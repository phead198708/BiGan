"""Phase 1.5 policy training runner tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
import xgboost as xgb

import bigan.v8.phase1.training as phase1_training
from bigan.v8.phase0 import (
    FEATURE_VECTOR_SCHEMA,
    LABEL_SCHEMA,
    MARKET_DATA_SCHEMA,
    DatasetContract,
    FeatureProvenance,
    FeatureVector,
    Label,
    Phase0Dataset,
    ValidationReport,
)
from bigan.v8.phase0.contracts import PHASE0_DATASET_VERSION
from bigan.v8.phase1 import (
    PHASE15_TRAINING_PHASE,
    PolicyAcceptanceConfig,
    PolicyDatasetConfig,
    PolicyTrainingRunConfig,
    XGBoostPolicyConfig,
    run_policy_training,
)
from bigan.v8.phase1.validation import PolicyAcceptanceFailure, PolicyAcceptanceReport

MINUTE_MS = 60_000


def _ts_at(year: int, month: int, day: int, hour: int, minute: int) -> int:
    return int(datetime(year, month, day, hour, minute, tzinfo=UTC).timestamp() * 1000)


def _phase0_training_dataset(row_count: int = 96) -> Phase0Dataset:
    t0 = _ts_at(2026, 6, 1, 12, 0)
    features: list[FeatureVector] = []
    labels: list[Label] = []
    for idx in range(row_count):
        decision_ts = t0 + idx * MINUTE_MS
        signal = -1.0 if idx % 4 in (0, 1) else 1.0
        shadow_return = 0.020 if signal > 0.0 else -0.015
        feature_values = {
            "signal": signal,
            "volatility_5m": 0.004 + (idx % 6) * 0.004,
            "volatility_15m": 0.010 + (idx % 5) * 0.003,
            "spread_bps": 1.0 + idx % 7,
            "liquidity_depth": 40.0 + (idx % 9) * 30.0,
        }
        provenance = {
            name: FeatureProvenance(
                feature_name=name,
                input_start_ts=decision_ts,
                input_end_ts=decision_ts,
                available_at_ts=decision_ts,
                lookback_ms=0,
            )
            for name in feature_values
        }
        features.append(
            FeatureVector(
                decision_ts=decision_ts,
                feature_cutoff_ts=decision_ts,
                lookback_start_ts=decision_ts,
                max_input_ts=decision_ts,
                source="polymarket",
                instrument_id="btc-updown-15m:UP",
                features=feature_values,
                provenance=provenance,
            )
        )
        labels.append(
            Label(
                decision_ts=decision_ts,
                label_ts=decision_ts + MINUTE_MS,
                horizon_ms=MINUTE_MS,
                source="polymarket",
                instrument_id="btc-updown-15m:UP",
                entry_price=100.0,
                exit_price=100.0 * (1.0 + shadow_return),
                gross_return=shadow_return,
                spread_cost=0.0,
                fee_cost=0.0,
                slippage_cost=0.0,
                liquidity_impact_cost=0.0,
                total_cost=0.0,
                net_return=shadow_return,
                is_positive=shadow_return > 0.0,
            )
        )

    dataset_hash = "phase0-policy-training-test-hash"
    contract = DatasetContract(
        dataset_version=PHASE0_DATASET_VERSION,
        dataset_hash=dataset_hash,
        market_schema=tuple(MARKET_DATA_SCHEMA.names),
        feature_schema=tuple(FEATURE_VECTOR_SCHEMA.names),
        label_schema=tuple(LABEL_SCHEMA.names),
        metadata={
            "market_rows": 0,
            "feature_rows": len(features),
            "label_rows": len(labels),
        },
    )
    validation_report = ValidationReport(metrics={"dataset_hash": dataset_hash})
    manifest = {
        "dataset_version": PHASE0_DATASET_VERSION,
        "dataset_hash": dataset_hash,
        "market_rows": 0,
        "feature_rows": len(features),
        "label_rows": len(labels),
        "feature_columns": list(FEATURE_VECTOR_SCHEMA.names),
        "dataset_contract": contract.to_dict(),
        "config": {"require_cost_calibration": False},
        "validation": validation_report.to_dict(),
    }
    return Phase0Dataset(
        market_data=[],
        features=features,
        labels=labels,
        validation_report=validation_report,
        manifest=manifest,
    )


def _run_config(output_dir: Path | str | None = None) -> PolicyTrainingRunConfig:
    return PolicyTrainingRunConfig(
        policy_dataset_config=PolicyDatasetConfig(
            horizon_ms=MINUTE_MS,
            feature_columns=(
                "signal",
                "volatility_5m",
                "volatility_15m",
                "spread_bps",
                "liquidity_depth",
            ),
        ),
        xgboost_config=XGBoostPolicyConfig(
            num_boost_round=8,
            max_depth=2,
            learning_rate=0.2,
            action_activation_threshold=0.50,
            seed=7,
        ),
        acceptance_config=PolicyAcceptanceConfig(
            min_active_regime_count=1,
            max_dominant_bucket_ratio=1.0,
            max_active_rate=1.0,
            min_action_std=0.0,
        ),
        train_fraction=0.65,
        output_dir=output_dir,
        created_at="2026-06-21T00:00:00Z",
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_policy_training_runner_writes_accepted_candidate_artifacts(tmp_path: Path) -> None:
    result = run_policy_training(
        _phase0_training_dataset(),
        _run_config(output_dir=str(tmp_path)),
    )

    assert result.accepted
    assert result.acceptance_report.passed
    assert result.run_manifest["phase"] == PHASE15_TRAINING_PHASE
    assert result.run_manifest["accepted"] is True
    assert result.run_manifest["candidate_status"] == "accepted"
    assert result.run_manifest["phase0_dataset_hash"] == result.policy_dataset.phase0_dataset_hash
    assert result.run_manifest["policy_dataset_hash"] == result.policy_dataset.policy_dataset_hash
    assert result.run_manifest["train_dataset_hash"] == result.split.train_dataset_hash
    assert result.run_manifest["shadow_dataset_hash"] == result.split.shadow_dataset_hash
    assert result.run_manifest["split_hash"] == result.split.split_hash
    assert result.run_manifest["direct_pnl_optimization"] is False
    assert result.run_manifest["shadow_return_used_for_training"] is False
    assert result.model.training_manifest["split"]["split_hash"] == result.split.split_hash
    assert result.artifact_dir is not None
    assert result.run_manifest["config"]["output_dir"] == str(tmp_path)

    expected_files = {
        "policy_dataset_manifest.json",
        "split_manifest.json",
        "training_manifest.json",
        "shadow_acceptance_report.json",
        "run_manifest.json",
        "model.xgb",
    }
    assert expected_files == {path.name for path in result.artifact_dir.iterdir()}
    saved_manifest = json.loads((result.artifact_dir / "run_manifest.json").read_text())
    assert saved_manifest["accepted"] is True
    artifacts = saved_manifest["artifacts"]
    assert artifacts["hash_algorithm"] == "sha256"
    assert artifacts["policy_dataset_manifest_sha256"] == _sha256_file(
        result.artifact_dir / artifacts["policy_dataset_manifest_path"]
    )
    assert artifacts["split_manifest_sha256"] == _sha256_file(
        result.artifact_dir / artifacts["split_manifest_path"]
    )
    assert artifacts["training_manifest_sha256"] == _sha256_file(
        result.artifact_dir / artifacts["training_manifest_path"]
    )
    assert artifacts["shadow_acceptance_report_sha256"] == _sha256_file(
        result.artifact_dir / artifacts["shadow_acceptance_report_path"]
    )
    assert artifacts["model_sha256"] == _sha256_file(
        result.artifact_dir / artifacts["model_path"]
    )
    canonical_manifest = saved_manifest.copy()
    canonical_manifest["artifacts"] = artifacts.copy()
    canonical_manifest["artifacts"]["run_manifest_canonical_sha256"] = ""
    canonical_digest = hashlib.sha256(
        json.dumps(
            canonical_manifest,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert artifacts["run_manifest_canonical_sha256"] == canonical_digest

    booster = xgb.Booster()
    booster.load_model(result.artifact_dir / "model.xgb")
    training_manifest_attr = booster.attr("training_manifest")
    assert training_manifest_attr is not None
    assert json.loads(training_manifest_attr)["split"]["split_hash"] == result.split.split_hash


def test_policy_training_registry_refuses_silent_overwrite_by_default(tmp_path: Path) -> None:
    phase0_dataset = _phase0_training_dataset()
    config = _run_config(output_dir=tmp_path)
    first = run_policy_training(phase0_dataset, config)
    assert first.artifact_dir is not None
    manifest_path = first.artifact_dir / "run_manifest.json"
    original_manifest_text = manifest_path.read_text(encoding="utf-8")

    with pytest.raises(FileExistsError, match="overwrite_existing=True"):
        run_policy_training(phase0_dataset, config)

    assert manifest_path.read_text(encoding="utf-8") == original_manifest_text

    overwrite_config = replace(
        config,
        created_at="2026-06-21T00:01:00Z",
        overwrite_existing=True,
    )
    overwritten = run_policy_training(phase0_dataset, overwrite_config)

    assert overwritten.artifact_dir == first.artifact_dir
    assert overwritten.run_manifest["run_id"] == first.run_manifest["run_id"]
    assert overwritten.run_manifest["created_at"] == "2026-06-21T00:01:00Z"
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["created_at"] == (
        "2026-06-21T00:01:00Z"
    )


def test_failed_shadow_acceptance_writes_rejected_candidate(tmp_path: Path) -> None:
    config = _run_config(output_dir=tmp_path)
    strict_config = PolicyTrainingRunConfig(
        policy_dataset_config=config.policy_dataset_config,
        xgboost_config=config.xgboost_config,
        acceptance_config=PolicyAcceptanceConfig(
            min_shadow_sharpe=1_000_000.0,
            min_active_regime_count=1,
            max_dominant_bucket_ratio=1.0,
            max_active_rate=1.0,
            min_action_std=0.0,
        ),
        train_fraction=config.train_fraction,
        output_dir=config.output_dir,
        created_at=config.created_at,
    )

    result = run_policy_training(_phase0_training_dataset(), strict_config)

    assert not result.accepted
    assert not result.acceptance_report.passed
    assert result.run_manifest["accepted"] is False
    assert result.run_manifest["candidate_status"] == "rejected"
    assert result.run_manifest["acceptance_report_passed"] is False
    assert "shadow_sharpe_non_positive" in result.run_manifest["acceptance_failure_codes"]
    assert result.artifact_dir is not None
    saved_manifest = json.loads((result.artifact_dir / "run_manifest.json").read_text())
    assert saved_manifest["accepted"] is False
    assert (result.artifact_dir / "model.xgb").exists()


def test_policy_training_runner_outputs_are_deterministic(tmp_path: Path) -> None:
    phase0_dataset = _phase0_training_dataset()
    first = run_policy_training(
        phase0_dataset,
        _run_config(output_dir=tmp_path / "first"),
    )
    second = run_policy_training(
        phase0_dataset,
        _run_config(output_dir=tmp_path / "second"),
    )

    assert first.policy_dataset.policy_dataset_hash == second.policy_dataset.policy_dataset_hash
    assert first.split.split_hash == second.split.split_hash
    assert first.run_manifest["run_id"] == second.run_manifest["run_id"]
    assert first.run_manifest["created_at"] == second.run_manifest["created_at"]
    assert first.run_manifest["direct_pnl_optimization"] is False
    assert second.run_manifest["direct_pnl_optimization"] is False


def test_split_provenance_failure_cannot_register_accepted_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    def mismatched_split_report(*args, **kwargs) -> PolicyAcceptanceReport:
        return PolicyAcceptanceReport(
            failures=(
                PolicyAcceptanceFailure(
                    code="split_hash_mismatch",
                    message="model training split_hash does not match the supplied split",
                ),
            ),
            metrics={
                "split_provenance": {
                    "passed": False,
                    "split_hash_matches": False,
                },
            },
            acceptance_criteria={
                "shadow_sharpe_positive": False,
                "stable_action_distribution": False,
                "monotonic_pnl_bucket_behavior": False,
                "regime_action_stability": False,
                "no_direct_pnl_optimization": True,
                "split_provenance_verified": False,
            },
        )

    monkeypatch.setattr(
        phase1_training,
        "validate_policy_shadow_split",
        mismatched_split_report,
    )

    result = run_policy_training(_phase0_training_dataset(), _run_config())

    assert not result.accepted
    assert result.run_manifest["accepted"] is False
    assert result.run_manifest["acceptance_failure_codes"] == ["split_hash_mismatch"]


def test_direct_pnl_policy_training_config_is_rejected_before_training() -> None:
    with pytest.raises(ValueError, match="PnL/profit"):
        XGBoostPolicyConfig(selection_metric="shadow_sharpe")


def test_phase0_gate_failure_stops_policy_training() -> None:
    phase0_dataset = _phase0_training_dataset()
    phase0_dataset.manifest["validation"]["acceptance_criteria"]["zero_detectable_leakage"] = False

    with pytest.raises(Exception, match="zero_detectable_leakage"):
        run_policy_training(phase0_dataset, _run_config())
