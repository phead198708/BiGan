"""Phase 1.5 candidate artifact loader for Phase 2."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import xgboost as xgb

from bigan.v8.phase1.contracts import XGBoostPolicyConfig
from bigan.v8.phase1.model import XGBoostPolicyModel
from bigan.v8.phase2.contracts import Phase2ArtifactError

REQUIRED_PHASE15_FILES: tuple[str, ...] = (
    "run_manifest.json",
    "training_manifest.json",
    "split_manifest.json",
    "shadow_acceptance_report.json",
    "model.xgb",
)

REQUIRED_RUN_HASH_FIELDS: tuple[str, ...] = (
    "policy_dataset_hash",
    "split_hash",
    "train_dataset_hash",
    "shadow_dataset_hash",
)


@dataclass(frozen=True, slots=True)
class Phase15CandidateArtifact:
    """Loaded immutable Phase 1.5 candidate consumed by Phase 2."""

    artifact_dir: Path
    run_manifest: dict[str, Any]
    training_manifest: dict[str, Any]
    split_manifest: dict[str, Any]
    shadow_acceptance_report: dict[str, Any]
    model_path: Path
    model_sha256: str
    model: XGBoostPolicyModel

    @property
    def run_id(self) -> str:
        return str(self.run_manifest["run_id"])

    @property
    def policy_dataset_hash(self) -> str:
        return str(self.run_manifest["policy_dataset_hash"])

    @property
    def split_hash(self) -> str:
        return str(self.run_manifest["split_hash"])

    @property
    def train_dataset_hash(self) -> str:
        return str(self.run_manifest["train_dataset_hash"])

    @property
    def shadow_dataset_hash(self) -> str:
        return str(self.run_manifest["shadow_dataset_hash"])

    def phase1_5_hashes(self) -> dict[str, str]:
        artifacts = self.run_manifest.get("artifacts", {})
        return {
            "run_id": self.run_id,
            "policy_dataset_hash": self.policy_dataset_hash,
            "split_hash": self.split_hash,
            "train_dataset_hash": self.train_dataset_hash,
            "shadow_dataset_hash": self.shadow_dataset_hash,
            "model_sha256": self.model_sha256,
            "training_manifest_sha256": str(artifacts.get("training_manifest_sha256", "")),
            "shadow_acceptance_report_sha256": str(
                artifacts.get("shadow_acceptance_report_sha256", "")
            ),
            "split_manifest_sha256": str(artifacts.get("split_manifest_sha256", "")),
            "run_manifest_canonical_sha256": str(
                artifacts.get("run_manifest_canonical_sha256", "")
            ),
        }


def load_phase15_candidate(artifact_dir: Path | str) -> Phase15CandidateArtifact:
    """Load and verify an accepted Phase 1.5 candidate artifact directory."""

    resolved_dir = Path(artifact_dir)
    _assert_required_files(resolved_dir)
    run_manifest = _read_json(resolved_dir / "run_manifest.json")
    training_manifest = _read_json(resolved_dir / "training_manifest.json")
    split_manifest = _read_json(resolved_dir / "split_manifest.json")
    shadow_acceptance_report = _read_json(resolved_dir / "shadow_acceptance_report.json")
    model_path = resolved_dir / "model.xgb"

    _validate_run_manifest(run_manifest)
    _validate_training_manifest(run_manifest, training_manifest)
    _validate_split_manifest(run_manifest, training_manifest, split_manifest)
    _validate_shadow_acceptance(run_manifest, shadow_acceptance_report)
    _validate_recorded_artifact_hashes(
        artifact_dir=resolved_dir,
        run_manifest=run_manifest,
        required_hash_names=("model_sha256",),
    )

    model_sha256 = str(run_manifest["artifacts"]["model_sha256"])
    model = _load_policy_model(
        model_path=model_path,
        training_manifest=training_manifest,
    )
    return Phase15CandidateArtifact(
        artifact_dir=resolved_dir,
        run_manifest=run_manifest,
        training_manifest=training_manifest,
        split_manifest=split_manifest,
        shadow_acceptance_report=shadow_acceptance_report,
        model_path=model_path,
        model_sha256=model_sha256,
        model=model,
    )


def _assert_required_files(artifact_dir: Path) -> None:
    if not artifact_dir.exists():
        raise Phase2ArtifactError(f"Phase 1.5 artifact directory does not exist: {artifact_dir}")
    missing = [name for name in REQUIRED_PHASE15_FILES if not (artifact_dir / name).is_file()]
    if missing:
        raise Phase2ArtifactError(
            "Phase 1.5 artifact directory is missing required files: "
            + ", ".join(sorted(missing))
        )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Phase2ArtifactError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise Phase2ArtifactError(f"JSON artifact must contain an object: {path}")
    return payload


def _validate_run_manifest(run_manifest: dict[str, Any]) -> None:
    if run_manifest.get("accepted") is not True:
        raise Phase2ArtifactError("Phase 2 requires an accepted Phase 1.5 candidate")
    if run_manifest.get("candidate_status") != "accepted":
        raise Phase2ArtifactError("Phase 1.5 candidate_status must be accepted")
    if run_manifest.get("acceptance_report_passed") is not True:
        raise Phase2ArtifactError("Phase 1.5 acceptance_report_passed must be true")
    if run_manifest.get("direct_pnl_optimization") is not False:
        raise Phase2ArtifactError("Phase 1.5 direct_pnl_optimization must be false")
    if run_manifest.get("shadow_return_used_for_training") is not False:
        raise Phase2ArtifactError("Phase 1.5 shadow_return_used_for_training must be false")
    if not str(run_manifest.get("run_id", "")).strip():
        raise Phase2ArtifactError("Phase 1.5 run_id is required")
    missing_hash_fields = [
        field_name
        for field_name in REQUIRED_RUN_HASH_FIELDS
        if not str(run_manifest.get(field_name, "")).strip()
    ]
    if missing_hash_fields:
        raise Phase2ArtifactError(
            "Phase 1.5 run_manifest is missing required hashes: "
            + ", ".join(missing_hash_fields)
        )
    artifacts = run_manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise Phase2ArtifactError("Phase 1.5 run_manifest artifacts block is required")
    if not str(artifacts.get("model_sha256", "")).strip():
        raise Phase2ArtifactError("Phase 1.5 model_sha256 must be recorded")


def _validate_training_manifest(
    run_manifest: dict[str, Any],
    training_manifest: dict[str, Any],
) -> None:
    if training_manifest.get("direct_pnl_optimization") is not False:
        raise Phase2ArtifactError("training_manifest direct_pnl_optimization must be false")
    if training_manifest.get("shadow_return_used_for_training") is not False:
        raise Phase2ArtifactError("training_manifest shadow_return_used_for_training must be false")
    if training_manifest.get("training_label_field") != run_manifest.get("training_label_field"):
        raise Phase2ArtifactError("training label field mismatch between run and training manifests")
    for field_name in (
        "policy_dataset_hash",
        "phase0_dataset_hash",
        "phase0_dataset_version",
        "train_dataset_hash",
        "shadow_dataset_hash",
    ):
        if training_manifest.get(field_name) != run_manifest.get(field_name):
            raise Phase2ArtifactError(f"training_manifest {field_name} mismatch")
    split_block = training_manifest.get("split")
    if not isinstance(split_block, Mapping):
        raise Phase2ArtifactError("training_manifest split block is required")
    if split_block.get("split_hash") != run_manifest.get("split_hash"):
        raise Phase2ArtifactError("training_manifest split_hash mismatch")


def _validate_split_manifest(
    run_manifest: dict[str, Any],
    training_manifest: dict[str, Any],
    split_manifest: dict[str, Any],
) -> None:
    for field_name in ("split_hash", "train_dataset_hash", "shadow_dataset_hash"):
        if split_manifest.get(field_name) != run_manifest.get(field_name):
            raise Phase2ArtifactError(f"split_manifest {field_name} mismatch")
    training_split = training_manifest.get("split")
    if not isinstance(training_split, Mapping):
        raise Phase2ArtifactError("training_manifest split block is required")
    for field_name in ("split_hash", "train_dataset_hash", "shadow_dataset_hash"):
        if split_manifest.get(field_name) != training_split.get(field_name):
            raise Phase2ArtifactError(f"split_manifest {field_name} training mismatch")


def _validate_shadow_acceptance(
    run_manifest: dict[str, Any],
    shadow_acceptance_report: dict[str, Any],
) -> None:
    if shadow_acceptance_report.get("passed") is not True:
        raise Phase2ArtifactError("shadow_acceptance_report passed must be true")
    criteria = shadow_acceptance_report.get("acceptance_criteria")
    if not isinstance(criteria, Mapping):
        raise Phase2ArtifactError("shadow_acceptance_report acceptance_criteria is required")
    if criteria.get("split_provenance_verified") is not True:
        raise Phase2ArtifactError("Phase 1.5 split_provenance_verified must be true")
    metrics = shadow_acceptance_report.get("metrics")
    if not isinstance(metrics, Mapping):
        raise Phase2ArtifactError("shadow_acceptance_report metrics are required")
    if metrics.get("split_hash") != run_manifest.get("split_hash"):
        raise Phase2ArtifactError("shadow_acceptance_report split_hash mismatch")


def _validate_recorded_artifact_hashes(
    *,
    artifact_dir: Path,
    run_manifest: dict[str, Any],
    required_hash_names: tuple[str, ...],
) -> None:
    artifacts = run_manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise Phase2ArtifactError("Phase 1.5 artifacts block is required")
    for hash_name in required_hash_names:
        if not str(artifacts.get(hash_name, "")).strip():
            raise Phase2ArtifactError(f"required artifact hash missing: {hash_name}")
    for path_key, hash_key in (
        ("model_path", "model_sha256"),
        ("training_manifest_path", "training_manifest_sha256"),
        ("shadow_acceptance_report_path", "shadow_acceptance_report_sha256"),
        ("split_manifest_path", "split_manifest_sha256"),
        ("policy_dataset_manifest_path", "policy_dataset_manifest_sha256"),
    ):
        path_value = artifacts.get(path_key)
        hash_value = artifacts.get(hash_key)
        if path_value is None or hash_value is None:
            continue
        path = artifact_dir / str(path_value)
        if not path.is_file():
            raise Phase2ArtifactError(f"recorded artifact path does not exist: {path_value}")
        actual = _sha256_file(path)
        if actual != hash_value:
            raise Phase2ArtifactError(f"artifact hash mismatch for {path_value}")


def _load_policy_model(
    *,
    model_path: Path,
    training_manifest: dict[str, Any],
) -> XGBoostPolicyModel:
    feature_columns = training_manifest.get("feature_columns")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise Phase2ArtifactError("training_manifest feature_columns must be a non-empty list")
    training_config = training_manifest.get("training_config")
    if not isinstance(training_config, Mapping):
        raise Phase2ArtifactError("training_manifest training_config is required")
    config_payload = dict(training_config)
    if "regime_feature_names" in config_payload:
        config_payload["regime_feature_names"] = tuple(config_payload["regime_feature_names"])
    try:
        config = XGBoostPolicyConfig(**config_payload)
    except (TypeError, ValueError) as exc:
        raise Phase2ArtifactError("invalid Phase 1.5 training_config") from exc
    booster = xgb.Booster()
    booster.load_model(model_path)
    return XGBoostPolicyModel(
        booster=booster,
        feature_columns=tuple(str(column) for column in feature_columns),
        config=config,
        training_manifest=training_manifest,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
