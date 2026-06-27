"""Real-corpus retraining gate for v8 Polymarket policy models."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import looks_like_sha256
from bigan.v8.polymarket.corpus import (
    POLYMARKET_SELL_BEFORE_CLOSE_LABEL_SCHEMA_VERSION,
)
from bigan.v8.polymarket.training.contracts import (
    DEFAULT_POLICY_CREATED_AT,
    PolymarketPolicyTrainingConfig,
    PolymarketPolicyTrainingResult,
    safety_fields,
)

POLYMARKET_REAL_CORPUS_RETRAINING_GATE_SCHEMA_VERSION = (
    "bigan-v8-polymarket-real-corpus-retraining-gate-v1"
)
POLYMARKET_REAL_CORPUS_RETRAINING_GATE_PHASE = "polymarket_real_corpus_retraining_gate"
DEFAULT_REAL_CORPUS_MODEL_VERSION = "polymarket_real_history_action_value_v1"
DEFAULT_REAL_CORPUS_TRAINING_RUN_ID = "polymarket_real_history_policy_run"


@dataclass(frozen=True, slots=True)
class PolymarketRealCorpusRetrainingGateConfig:
    """Configuration for the recorder-bundle to policy-training hard gate."""

    recorder_run_dir: Path | str
    output_dir: Path | str
    run_id: str = "polymarket_real_corpus_retraining_gate"
    training_run_id: str = DEFAULT_REAL_CORPUS_TRAINING_RUN_ID
    model_version: str = DEFAULT_REAL_CORPUS_MODEL_VERSION
    created_at: str = DEFAULT_POLICY_CREATED_AT
    train_fraction: float = 0.60
    validation_fraction: float = 0.25
    ev_threshold: float = 0.015
    min_confidence: float = 0.05
    max_paper_notional: float = 0.20
    fee_rate: float = 0.0002
    slippage_rate: float = 0.0005
    liquidity_impact_rate: float = 0.0001
    overwrite_existing: bool = False
    paper_only: bool = True
    capital_at_risk: bool = False
    broker_exchange_write_enabled: bool = False
    live_exchange_write_enabled: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.recorder_run_dir, Path):
            object.__setattr__(self, "recorder_run_dir", Path(self.recorder_run_dir))
        if not isinstance(self.output_dir, Path):
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if not self.training_run_id.strip():
            raise ValueError("training_run_id is required")
        if not self.model_version.strip():
            raise ValueError("model_version is required")
        if not self.created_at:
            raise ValueError("created_at is required")
        for field_name, expected in safety_fields().items():
            if getattr(self, field_name) is not expected:
                raise ValueError(f"{field_name} must be {str(expected).lower()}")

    @property
    def run_dir(self) -> Path:
        return self.output_dir.expanduser().resolve() / self.run_id

    @property
    def training_output_dir(self) -> Path:
        return self.run_dir / "policy_training"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["recorder_run_dir"] = str(self.recorder_run_dir)
        payload["output_dir"] = str(self.output_dir)
        return payload


@dataclass(frozen=True, slots=True)
class PolymarketRealCorpusRetrainingGateResult:
    """Output handles for one real-corpus retraining gate run."""

    run_dir: Path
    artifact_paths: dict[str, Path]
    artifact_hashes: dict[str, str]
    report: dict[str, Any]
    manifest: dict[str, Any]
    model_manifest: dict[str, Any] | None
    training_result: PolymarketPolicyTrainingResult | None


def run_polymarket_real_corpus_retraining_gate(
    config: PolymarketRealCorpusRetrainingGateConfig,
) -> PolymarketRealCorpusRetrainingGateResult:
    """Validate a real recorder bundle and run #133 policy training if eligible."""

    run_dir = config.run_dir
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(
                f"real corpus retraining gate run_dir already exists: {run_dir}; "
                "set overwrite_existing=True to replace it"
            )
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    artifact_paths = {
        "gate_report": run_dir / "real_corpus_retraining_gate_report.json",
        "gate_manifest": run_dir / "real_corpus_training_manifest.json",
    }
    recorder_evidence = _load_recorder_evidence(config.recorder_run_dir)
    reason_codes = _recorder_retraining_reason_codes(recorder_evidence)
    training_result: PolymarketPolicyTrainingResult | None = None
    model_manifest: dict[str, Any] | None = None

    if not reason_codes:
        training_result = _run_real_corpus_training(config, recorder_evidence)
        model_manifest = _augment_model_manifest(
            config=config,
            recorder_evidence=recorder_evidence,
            training_result=training_result,
        )
        _write_json(training_result.artifact_paths["model_manifest"], model_manifest)
        artifact_paths.update(
            {f"policy_{name}": path for name, path in training_result.artifact_paths.items()}
        )

    report = _gate_report(
        config=config,
        recorder_evidence=recorder_evidence,
        reason_codes=reason_codes,
        training_result=training_result,
        model_manifest=model_manifest,
    )
    manifest = _gate_manifest(
        config=config,
        report=report,
        recorder_evidence=recorder_evidence,
        model_manifest=model_manifest,
    )
    _write_json(artifact_paths["gate_report"], report)
    _write_json(artifact_paths["gate_manifest"], manifest)
    artifact_hashes = {
        name: _sha256_file(path) for name, path in sorted(artifact_paths.items()) if path.exists()
    }
    return PolymarketRealCorpusRetrainingGateResult(
        run_dir=run_dir,
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
        report=report,
        manifest=manifest,
        model_manifest=model_manifest,
        training_result=training_result,
    )


def _run_real_corpus_training(
    config: PolymarketRealCorpusRetrainingGateConfig,
    recorder_evidence: dict[str, Any],
) -> PolymarketPolicyTrainingResult:
    from bigan.v8.polymarket.training.runner import run_polymarket_policy_training

    return run_polymarket_policy_training(
        PolymarketPolicyTrainingConfig(
            corpus_dir=recorder_evidence["phase2_corpus_dir"],
            output_dir=config.training_output_dir,
            run_id=config.training_run_id,
            model_version=config.model_version,
            created_at=config.created_at,
            train_fraction=config.train_fraction,
            validation_fraction=config.validation_fraction,
            ev_threshold=config.ev_threshold,
            min_confidence=config.min_confidence,
            max_paper_notional=config.max_paper_notional,
            fee_rate=config.fee_rate,
            slippage_rate=config.slippage_rate,
            liquidity_impact_rate=config.liquidity_impact_rate,
            overwrite_existing=True,
        )
    )


def _augment_model_manifest(
    *,
    config: PolymarketRealCorpusRetrainingGateConfig,
    recorder_evidence: dict[str, Any],
    training_result: PolymarketPolicyTrainingResult,
) -> dict[str, Any]:
    split = recorder_evidence["phase2_split"]
    model_manifest = dict(training_result.model_manifest)
    model_manifest.update(
        {
            "real_historical_corpus_used": True,
            "fixture_corpus_used": False,
            "synthetic_corpus_used": False,
            "synthetic_fixture_signal_used": False,
            "fixture_model_used": False,
            "manual_live_evidence_eligible": True,
            "recorder_run_id": recorder_evidence["recorder_report"]["run_id"],
            "recorder_manifest_sha256": recorder_evidence["recorder_manifest_sha256"],
            "recorder_report_sha256": recorder_evidence["recorder_report_sha256"],
            "phase2_corpus_manifest_sha256": recorder_evidence[
                "phase2_corpus_manifest_sha256"
            ],
            "phase2_train_shadow_split_sha256": recorder_evidence["phase2_split_sha256"],
            "split_hash": split["split_hash"],
            "train_dataset_hash": split["train_dataset_hash"],
            "shadow_dataset_hash": split["shadow_dataset_hash"],
            "policy_dataset_hash": model_manifest["dataset_hash"],
            "real_corpus_gate_run_id": config.run_id,
            "real_corpus_identity": {
                "recorder_run_id": recorder_evidence["recorder_report"]["run_id"],
                "public_collection_status": recorder_evidence["recorder_report"][
                    "public_collection_status"
                ],
                "live_polymarket_data_read": recorder_evidence["recorder_report"][
                    "live_polymarket_data_read"
                ],
                "live_btc_reference_data_read": recorder_evidence["recorder_report"][
                    "live_btc_reference_data_read"
                ],
                "raw_artifact_hashes": recorder_evidence["recorder_manifest"][
                    "raw_artifact_hashes"
                ],
                "raw_artifact_row_counts": recorder_evidence["recorder_manifest"][
                    "raw_artifact_row_counts"
                ],
            },
        }
    )
    _assert_augmented_manifest_hashes(model_manifest)
    return model_manifest


def _load_recorder_evidence(recorder_run_dir: Path | str) -> dict[str, Any]:
    run_dir = Path(recorder_run_dir).expanduser().resolve()
    recorder_manifest_path = run_dir / "real_corpus_recorder_manifest.json"
    recorder_report_path = run_dir / "real_corpus_recorder_report.json"
    evidence: dict[str, Any] = {
        "recorder_run_dir": run_dir,
        "recorder_manifest_path": recorder_manifest_path,
        "recorder_report_path": recorder_report_path,
        "missing_paths": [],
    }
    if not recorder_manifest_path.exists():
        evidence["missing_paths"].append("real_corpus_recorder_manifest.json")
    if not recorder_report_path.exists():
        evidence["missing_paths"].append("real_corpus_recorder_report.json")
    if evidence["missing_paths"]:
        return evidence

    recorder_manifest = _read_json(recorder_manifest_path)
    recorder_report = _read_json(recorder_report_path)
    phase2_corpus_dir = Path(
        str(recorder_manifest.get("phase2_corpus_dir") or run_dir / "phase2_corpus")
    ).expanduser()
    if not phase2_corpus_dir.is_absolute():
        phase2_corpus_dir = run_dir / phase2_corpus_dir
    phase2_corpus_dir = phase2_corpus_dir.resolve()
    corpus_manifest_path = phase2_corpus_dir / "polymarket_corpus_manifest.json"
    split_path = phase2_corpus_dir / "polymarket_train_shadow_split.json"
    evidence.update(
        {
            "recorder_manifest": recorder_manifest,
            "recorder_report": recorder_report,
            "recorder_manifest_sha256": _sha256_file(recorder_manifest_path),
            "recorder_report_sha256": _sha256_file(recorder_report_path),
            "phase2_corpus_dir": phase2_corpus_dir,
            "phase2_corpus_manifest_path": corpus_manifest_path,
            "phase2_split_path": split_path,
        }
    )
    if corpus_manifest_path.exists():
        evidence["phase2_corpus_manifest"] = _read_json(corpus_manifest_path)
        evidence["phase2_corpus_manifest_sha256"] = _sha256_file(corpus_manifest_path)
    if split_path.exists():
        evidence["phase2_split"] = _read_json(split_path)
        evidence["phase2_split_sha256"] = _sha256_file(split_path)
    return evidence


def _recorder_retraining_reason_codes(evidence: dict[str, Any]) -> list[str]:
    reasons: set[str] = set()
    for missing in evidence.get("missing_paths", []):
        reasons.add(f"missing_{missing.removesuffix('.json')}")
    if reasons:
        return sorted(reasons)

    report = evidence["recorder_report"]
    manifest = evidence["recorder_manifest"]
    required_true = {
        "phase2_corpus_build_eligible": "recorder_phase2_corpus_not_build_eligible",
        "real_historical_training_eligible": "recorder_real_historical_training_not_eligible",
        "manual_live_evidence_eligible": "recorder_manual_live_evidence_not_eligible",
        "real_historical_corpus_used": "recorder_real_historical_corpus_not_used",
        "live_polymarket_data_read": "recorder_live_polymarket_data_not_read",
        "live_btc_reference_data_read": "recorder_live_btc_reference_data_not_read",
        "phase2_corpus_built": "recorder_phase2_corpus_not_built",
    }
    required_false = {
        "mock_public_data_used": "recorder_mock_public_data_used",
        "synthetic_public_data_used": "recorder_synthetic_public_data_used",
        "synthetic_corpus_used": "recorder_synthetic_corpus_used",
        "fixture_corpus_used": "recorder_fixture_corpus_used",
    }
    for field_name, reason in required_true.items():
        if report.get(field_name) is not True:
            reasons.add(reason)
    for field_name, reason in required_false.items():
        if report.get(field_name) is not False:
            reasons.add(reason)
    if report.get("public_collection_status") != "completed":
        reasons.add("recorder_public_collection_not_completed")
    for field_name, expected in safety_fields().items():
        if report.get(field_name) is not expected or manifest.get(field_name) is not expected:
            reasons.add(f"unsafe_{field_name}")

    expected_corpus_manifest = str(report.get("phase2_corpus_manifest_sha256") or "")
    actual_corpus_manifest = str(evidence.get("phase2_corpus_manifest_sha256") or "")
    if not looks_like_sha256(expected_corpus_manifest):
        reasons.add("invalid_phase2_corpus_manifest_sha256")
    elif actual_corpus_manifest != expected_corpus_manifest:
        reasons.add("phase2_corpus_manifest_hash_mismatch")
    if "phase2_corpus_manifest" not in evidence:
        reasons.add("missing_phase2_corpus_manifest")
    if "phase2_split" not in evidence:
        reasons.add("missing_phase2_train_shadow_split")
    else:
        for field_name in ("split_hash", "train_dataset_hash", "shadow_dataset_hash"):
            if not looks_like_sha256(str(evidence["phase2_split"].get(field_name, ""))):
                reasons.add(f"invalid_{field_name}")
    corpus_manifest = evidence.get("phase2_corpus_manifest", {})
    recorder_raw_hashes = manifest.get("raw_artifact_hashes", {})
    if corpus_manifest and corpus_manifest.get("raw_artifact_hashes") != recorder_raw_hashes:
        reasons.add("phase2_raw_artifact_hash_mismatch")
    if corpus_manifest:
        if corpus_manifest.get("sell_before_close_label_schema_version") != (
            POLYMARKET_SELL_BEFORE_CLOSE_LABEL_SCHEMA_VERSION
        ):
            reasons.add("missing_executable_sell_before_close_label_schema")
        if corpus_manifest.get("sell_before_close_fixed_terminal_bid_only_labels_allowed") is True:
            reasons.add("fixed_terminal_bid_only_sell_before_close_labels_allowed")
        if corpus_manifest.get("sell_before_close_label_gate_passed") is not True:
            reasons.add("sell_before_close_label_redesign_gate_failed")
    return sorted(reasons)


def _gate_report(
    *,
    config: PolymarketRealCorpusRetrainingGateConfig,
    recorder_evidence: dict[str, Any],
    reason_codes: list[str],
    training_result: PolymarketPolicyTrainingResult | None,
    model_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    accepted = not reason_codes and training_result is not None and model_manifest is not None
    return {
        "schema_version": POLYMARKET_REAL_CORPUS_RETRAINING_GATE_SCHEMA_VERSION,
        "phase": POLYMARKET_REAL_CORPUS_RETRAINING_GATE_PHASE,
        "run_id": config.run_id,
        "created_at": config.created_at,
        "accepted_for_retraining": accepted,
        "training_completed": training_result is not None,
        "gate_status": "completed" if accepted else "blocked_fail_closed",
        "reason_codes": reason_codes,
        "recorder_run_dir": str(recorder_evidence["recorder_run_dir"]),
        "recorder_run_id": recorder_evidence.get("recorder_report", {}).get("run_id"),
        "recorder_manifest_sha256": recorder_evidence.get("recorder_manifest_sha256"),
        "recorder_report_sha256": recorder_evidence.get("recorder_report_sha256"),
        "phase2_corpus_dir": str(recorder_evidence.get("phase2_corpus_dir", "")),
        "phase2_corpus_manifest_sha256": recorder_evidence.get(
            "phase2_corpus_manifest_sha256"
        ),
        "model_manifest_path": None
        if training_result is None
        else str(training_result.artifact_paths["model_manifest"]),
        "model_sha256": None if model_manifest is None else model_manifest["model_sha256"],
        "model_manifest_sha256": None
        if training_result is None
        else _sha256_file(training_result.artifact_paths["model_manifest"]),
        "policy_dataset_hash": None
        if model_manifest is None
        else model_manifest["policy_dataset_hash"],
        "split_hash": None if model_manifest is None else model_manifest["split_hash"],
        "train_dataset_hash": None
        if model_manifest is None
        else model_manifest["train_dataset_hash"],
        "shadow_dataset_hash": None
        if model_manifest is None
        else model_manifest["shadow_dataset_hash"],
        "real_historical_corpus_used": accepted,
        "fixture_corpus_used": False,
        "synthetic_corpus_used": False,
        "synthetic_fixture_signal_used": False,
        "fixture_model_used": False,
        "manual_live_evidence_eligible": accepted,
        **safety_fields(),
    }


def _gate_manifest(
    *,
    config: PolymarketRealCorpusRetrainingGateConfig,
    report: dict[str, Any],
    recorder_evidence: dict[str, Any],
    model_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": POLYMARKET_REAL_CORPUS_RETRAINING_GATE_SCHEMA_VERSION,
        "phase": POLYMARKET_REAL_CORPUS_RETRAINING_GATE_PHASE,
        "run_id": config.run_id,
        "created_at": config.created_at,
        "config": config.to_dict(),
        "accepted_for_retraining": report["accepted_for_retraining"],
        "gate_status": report["gate_status"],
        "reason_codes": report["reason_codes"],
        "recorder_manifest_sha256": report["recorder_manifest_sha256"],
        "recorder_report_sha256": report["recorder_report_sha256"],
        "phase2_corpus_manifest_sha256": report["phase2_corpus_manifest_sha256"],
        "model_manifest_sha256": report["model_manifest_sha256"],
        "model_sha256": report["model_sha256"],
        "policy_dataset_hash": report["policy_dataset_hash"],
        "split_hash": report["split_hash"],
        "train_dataset_hash": report["train_dataset_hash"],
        "shadow_dataset_hash": report["shadow_dataset_hash"],
        "model_manifest": model_manifest,
        "recorder_raw_artifact_hashes": recorder_evidence.get("recorder_manifest", {}).get(
            "raw_artifact_hashes"
        ),
        "recorder_raw_artifact_row_counts": recorder_evidence.get(
            "recorder_manifest", {}
        ).get("raw_artifact_row_counts"),
        "real_historical_corpus_used": report["real_historical_corpus_used"],
        "fixture_corpus_used": report["fixture_corpus_used"],
        "synthetic_corpus_used": report["synthetic_corpus_used"],
        "synthetic_fixture_signal_used": report["synthetic_fixture_signal_used"],
        "fixture_model_used": report["fixture_model_used"],
        "manual_live_evidence_eligible": report["manual_live_evidence_eligible"],
        **safety_fields(),
    }


def _assert_augmented_manifest_hashes(model_manifest: dict[str, Any]) -> None:
    for field_name in (
        "model_sha256",
        "training_corpus_hash",
        "dataset_hash",
        "policy_dataset_hash",
        "split_hash",
        "train_dataset_hash",
        "shadow_dataset_hash",
        "recorder_manifest_sha256",
        "recorder_report_sha256",
        "phase2_corpus_manifest_sha256",
        "phase2_train_shadow_split_sha256",
    ):
        if not looks_like_sha256(str(model_manifest.get(field_name, ""))):
            raise ValueError(f"{field_name} must be SHA-256")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    if isinstance(value, Path):
        return str(value)
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
