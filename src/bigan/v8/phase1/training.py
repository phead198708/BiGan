"""Phase 1.5 policy training runner and local candidate registry."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bigan.v8.phase0.pipeline import Phase0Dataset
from bigan.v8.phase1.contracts import (
    PHASE1_POLICY_VERSION,
    PolicyDataset,
    PolicyDatasetConfig,
    PolicyTrainShadowSplit,
    XGBoostPolicyConfig,
)
from bigan.v8.phase1.dataset import (
    build_policy_dataset_from_phase0,
    build_temporal_policy_split,
)
from bigan.v8.phase1.model import XGBoostPolicyModel, train_xgboost_policy
from bigan.v8.phase1.validation import (
    PolicyAcceptanceConfig,
    PolicyAcceptanceReport,
    validate_policy_shadow_split,
)

PHASE15_TRAINING_PHASE = "phase1.5_policy_training"
DEFAULT_CREATED_AT = "1970-01-01T00:00:00Z"


@dataclass(frozen=True, slots=True)
class PolicyTrainingRunConfig:
    """Configuration for the Phase 1.5 policy training runner."""

    policy_dataset_config: PolicyDatasetConfig = PolicyDatasetConfig()
    xgboost_config: XGBoostPolicyConfig = XGBoostPolicyConfig()
    acceptance_config: PolicyAcceptanceConfig = PolicyAcceptanceConfig()
    train_fraction: float = 0.70
    output_dir: Path | None = None
    run_id: str | None = None
    created_at: str = DEFAULT_CREATED_AT

    def __post_init__(self) -> None:
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be in (0, 1)")
        if self.run_id is not None and not self.run_id.strip():
            raise ValueError("run_id must be non-empty when provided")
        if not self.created_at:
            raise ValueError("created_at is required")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["policy_dataset_config"] = self.policy_dataset_config.to_dict()
        payload["xgboost_config"] = self.xgboost_config.to_dict()
        payload["acceptance_config"] = self.acceptance_config.to_dict()
        payload["output_dir"] = None if self.output_dir is None else str(self.output_dir)
        return payload


@dataclass(frozen=True, slots=True)
class PolicyTrainingRunResult:
    """Result of a Phase 1.5 policy training run."""

    policy_dataset: PolicyDataset
    split: PolicyTrainShadowSplit
    model: XGBoostPolicyModel
    acceptance_report: PolicyAcceptanceReport
    run_manifest: dict[str, Any]
    artifact_dir: Path | None = None

    @property
    def accepted(self) -> bool:
        return bool(self.run_manifest["accepted"])


def run_policy_training(
    phase0_dataset: Phase0Dataset,
    config: PolicyTrainingRunConfig | None = None,
) -> PolicyTrainingRunResult:
    """Run the full Phase 1.5 candidate policy training workflow."""

    resolved_config = config or PolicyTrainingRunConfig()
    policy_dataset = build_policy_dataset_from_phase0(
        phase0_dataset,
        resolved_config.policy_dataset_config,
    )
    split = build_temporal_policy_split(
        policy_dataset,
        train_fraction=resolved_config.train_fraction,
    )
    model = train_xgboost_policy(
        policy_dataset,
        resolved_config.xgboost_config,
        split=split,
    )
    acceptance_report = validate_policy_shadow_split(
        model,
        split,
        resolved_config.acceptance_config,
    )
    run_manifest = _run_manifest(
        policy_dataset=policy_dataset,
        split=split,
        model=model,
        acceptance_report=acceptance_report,
        config=resolved_config,
    )
    artifact_dir = None
    if resolved_config.output_dir is not None:
        artifact_dir = _write_run_artifacts(
            output_dir=resolved_config.output_dir,
            run_id=str(run_manifest["run_id"]),
            policy_dataset=policy_dataset,
            split=split,
            model=model,
            acceptance_report=acceptance_report,
            run_manifest=run_manifest,
        )
    return PolicyTrainingRunResult(
        policy_dataset=policy_dataset,
        split=split,
        model=model,
        acceptance_report=acceptance_report,
        run_manifest=run_manifest,
        artifact_dir=artifact_dir,
    )


def _run_manifest(
    *,
    policy_dataset: PolicyDataset,
    split: PolicyTrainShadowSplit,
    model: XGBoostPolicyModel,
    acceptance_report: PolicyAcceptanceReport,
    config: PolicyTrainingRunConfig,
) -> dict[str, Any]:
    accepted = acceptance_report.passed
    run_id = config.run_id or _deterministic_run_id(
        policy_dataset=policy_dataset,
        split=split,
        model=model,
        config=config,
    )
    return {
        "phase": PHASE15_TRAINING_PHASE,
        "phase1_policy_version": PHASE1_POLICY_VERSION,
        "run_id": run_id,
        "accepted": accepted,
        "candidate_status": "accepted" if accepted else "rejected",
        "phase0_dataset_hash": policy_dataset.phase0_dataset_hash,
        "phase0_dataset_version": policy_dataset.phase0_dataset_version,
        "policy_dataset_hash": policy_dataset.policy_dataset_hash,
        "train_dataset_hash": split.train_dataset_hash,
        "shadow_dataset_hash": split.shadow_dataset_hash,
        "split_hash": split.split_hash,
        "train_row_count": len(split.train_examples),
        "shadow_row_count": len(split.shadow_examples),
        "model_version": model.config.model_version,
        "objective": model.config.objective,
        "target_encoding": policy_dataset.config.target_encoding,
        "direct_pnl_optimization": bool(model.training_manifest["direct_pnl_optimization"]),
        "shadow_return_used_for_training": bool(
            model.training_manifest["shadow_return_used_for_training"]
        ),
        "training_label_field": model.training_manifest["training_label_field"],
        "acceptance_report_passed": acceptance_report.passed,
        "acceptance_failure_codes": [
            failure.code for failure in acceptance_report.failures
        ],
        "created_at": config.created_at,
        "config": config.to_dict(),
    }


def _deterministic_run_id(
    *,
    policy_dataset: PolicyDataset,
    split: PolicyTrainShadowSplit,
    model: XGBoostPolicyModel,
    config: PolicyTrainingRunConfig,
) -> str:
    payload = {
        "phase": PHASE15_TRAINING_PHASE,
        "policy_dataset_hash": policy_dataset.policy_dataset_hash,
        "split_hash": split.split_hash,
        "model_config": model.config.to_dict(),
        "policy_dataset_config": policy_dataset.config.to_dict(),
        "acceptance_config": config.acceptance_config.to_dict(),
        "train_fraction": config.train_fraction,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return "phase1_5_" + hashlib.sha256(encoded).hexdigest()[:16]


def _write_run_artifacts(
    *,
    output_dir: Path,
    run_id: str,
    policy_dataset: PolicyDataset,
    split: PolicyTrainShadowSplit,
    model: XGBoostPolicyModel,
    acceptance_report: PolicyAcceptanceReport,
    run_manifest: dict[str, Any],
) -> Path:
    artifact_dir = output_dir / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _write_json(artifact_dir / "policy_dataset_manifest.json", policy_dataset.to_dict())
    _write_json(artifact_dir / "split_manifest.json", split.to_dict())
    _write_json(artifact_dir / "training_manifest.json", model.training_manifest)
    _write_json(artifact_dir / "shadow_acceptance_report.json", acceptance_report.to_dict())
    _write_json(artifact_dir / "run_manifest.json", run_manifest)
    model.booster.set_attr(
        training_manifest=json.dumps(
            _json_ready(model.training_manifest),
            sort_keys=True,
            separators=(",", ":"),
        ),
        run_manifest=json.dumps(
            _json_ready(run_manifest),
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    model.booster.save_model(artifact_dir / "model.xgb")
    return artifact_dir


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return value
