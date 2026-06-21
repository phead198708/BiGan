"""Phase 2 hybrid PnL-aware evaluation tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

import bigan.v8.phase1.model as phase1_model
import bigan.v8.phase1.training as phase1_training
import bigan.v8.phase2.evaluation as phase2_evaluation
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
from bigan.v8.phase0.costs import CostModelConfig
from bigan.v8.phase1 import (
    PolicyAcceptanceConfig,
    PolicyDatasetConfig,
    PolicyPrediction,
    PolicyTrainingExample,
    PolicyTrainingRunConfig,
    PolicyTrainShadowSplit,
    XGBoostPolicyConfig,
    run_policy_training,
)
from bigan.v8.phase1.model import XGBoostPolicyModel
from bigan.v8.phase2 import (
    PHASE2_EVALUATION_PHASE,
    ExecutionFill,
    ExecutionSimulationConfig,
    Phase2ArtifactError,
    Phase2EvaluationConfig,
    load_phase15_candidate,
    run_phase2_evaluation,
)

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
            "mid_price": 100.0 + idx * 0.01,
            "spread": 0.02 + 0.001 * (idx % 4),
            "spread_bps": 0.0002 + 0.00001 * (idx % 4),
            "volatility_5m": 0.004 + (idx % 6) * 0.001,
            "volatility_15m": 0.010 + (idx % 5) * 0.001,
            "volume_1m": 10.0 + idx % 5,
            "trade_count_1m": 1 + idx % 3,
            "liquidity_depth": 140.0 + (idx % 9) * 20.0,
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

    dataset_hash = "phase0-phase2-test-hash"
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


def _training_config(output_dir: Path) -> PolicyTrainingRunConfig:
    return PolicyTrainingRunConfig(
        policy_dataset_config=PolicyDatasetConfig(
            horizon_ms=MINUTE_MS,
            feature_columns=(
                "signal",
                "mid_price",
                "spread",
                "spread_bps",
                "volatility_5m",
                "volatility_15m",
                "volume_1m",
                "trade_count_1m",
                "liquidity_depth",
            ),
        ),
        xgboost_config=XGBoostPolicyConfig(
            num_boost_round=8,
            max_depth=2,
            learning_rate=0.2,
            action_activation_threshold=0.50,
            seed=11,
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


def _accepted_phase15_candidate(tmp_path: Path):
    return run_policy_training(
        _phase0_training_dataset(),
        _training_config(tmp_path / "phase15"),
    )


def _phase2_config(output_dir: Path | None = None) -> Phase2EvaluationConfig:
    return Phase2EvaluationConfig(
        execution_config=ExecutionSimulationConfig(
            cost_model_config=CostModelConfig(
                fee_bps=0.1,
                base_slippage_bps=0.1,
                volatility_slippage_factor=0.001,
                liquidity_impact_factor=0.0001,
            ),
            risk_penalty_factor=0.0,
            pnl_lambda=0.25,
            policy_edge_scale=0.20,
        ),
        min_sharpe_improvement_ratio=-10.0,
        min_turnover_reduction_ratio=-10.0,
        output_dir=output_dir,
        created_at="2026-06-21T00:05:00Z",
    )


def _shadow_example_without_spread() -> PolicyTrainingExample:
    return PolicyTrainingExample(
        decision_ts=_ts_at(2026, 6, 1, 12, 0),
        source="polymarket",
        instrument_id="btc-updown-15m:UP",
        features={
            "mid_price": 100.0,
            "spread_bps": 5.0,
            "volatility_5m": 0.0,
            "liquidity_depth": 1_000_000.0,
        },
        target_label=1.0,
        shadow_net_return=0.01,
        horizon_ms=MINUTE_MS,
        regime_key="polymarket|btc-updown-15m:UP|vol=0|spread=0|liq=2",
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _refresh_artifact_hashes(artifact_dir: Path) -> None:
    manifest_path = artifact_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest["artifacts"]
    for path_key, hash_key in (
        ("model_path", "model_sha256"),
        ("training_manifest_path", "training_manifest_sha256"),
        ("shadow_acceptance_report_path", "shadow_acceptance_report_sha256"),
        ("split_manifest_path", "split_manifest_sha256"),
        ("policy_dataset_manifest_path", "policy_dataset_manifest_sha256"),
    ):
        artifacts[hash_key] = _sha256_file(artifact_dir / artifacts[path_key])
    artifacts["run_manifest_canonical_sha256"] = ""
    artifacts["run_manifest_canonical_sha256"] = _canonical_run_manifest_hash(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _canonical_run_manifest_hash(manifest: dict) -> str:
    payload = json.loads(json.dumps(manifest, sort_keys=True, allow_nan=False))
    payload["artifacts"]["run_manifest_canonical_sha256"] = ""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_phase2_loader_verifies_accepted_candidate_artifacts(tmp_path: Path) -> None:
    phase15 = _accepted_phase15_candidate(tmp_path)
    assert phase15.artifact_dir is not None

    candidate = load_phase15_candidate(phase15.artifact_dir)

    assert candidate.run_id == phase15.run_manifest["run_id"]
    assert candidate.model_path.name == "model.xgb"
    assert candidate.model_sha256 == phase15.run_manifest["artifacts"]["model_sha256"]
    assert candidate.policy_dataset_hash == phase15.run_manifest["policy_dataset_hash"]
    assert candidate.split_hash == phase15.split.split_hash
    assert candidate.model.training_manifest["direct_pnl_optimization"] is False
    assert candidate.shadow_baseline_metrics["shadow_sharpe"] == (
        phase15.acceptance_report.metrics["shadow_sharpe"]
    )
    assert candidate.shadow_acceptance_report["acceptance_criteria"][
        "split_provenance_verified"
    ] is True


def test_phase2_rejects_rejected_phase15_candidate(tmp_path: Path) -> None:
    config = _training_config(tmp_path / "phase15")
    rejected_config = replace(
        config,
        acceptance_config=PolicyAcceptanceConfig(
            min_shadow_sharpe=1_000_000.0,
            min_active_regime_count=1,
            max_dominant_bucket_ratio=1.0,
            max_active_rate=1.0,
            min_action_std=0.0,
        ),
    )
    phase15 = run_policy_training(_phase0_training_dataset(), rejected_config)
    assert phase15.artifact_dir is not None

    with pytest.raises(Phase2ArtifactError, match="accepted Phase 1.5 candidate"):
        load_phase15_candidate(phase15.artifact_dir)


def test_phase2_rejects_missing_model_sha256(tmp_path: Path) -> None:
    phase15 = _accepted_phase15_candidate(tmp_path)
    assert phase15.artifact_dir is not None
    manifest_path = phase15.artifact_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["artifacts"]["model_sha256"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(Phase2ArtifactError, match="model_sha256"):
        load_phase15_candidate(phase15.artifact_dir)


def test_phase2_requires_all_recorded_hashes(tmp_path: Path) -> None:
    phase15 = _accepted_phase15_candidate(tmp_path)
    assert phase15.artifact_dir is not None
    manifest_path = phase15.artifact_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["artifacts"]["training_manifest_sha256"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(Phase2ArtifactError, match="training_manifest_sha256"):
        load_phase15_candidate(phase15.artifact_dir)


def test_phase2_detects_run_manifest_canonical_hash_mismatch(tmp_path: Path) -> None:
    phase15 = _accepted_phase15_candidate(tmp_path)
    assert phase15.artifact_dir is not None
    manifest_path = phase15.artifact_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["created_at"] = "2026-06-21T00:09:00Z"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(Phase2ArtifactError, match="run_manifest_canonical_sha256"):
        load_phase15_candidate(phase15.artifact_dir)


def test_phase2_rejects_direct_pnl_training_flag(tmp_path: Path) -> None:
    phase15 = _accepted_phase15_candidate(tmp_path)
    assert phase15.artifact_dir is not None
    manifest_path = phase15.artifact_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["direct_pnl_optimization"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(Phase2ArtifactError, match="direct_pnl_optimization"):
        load_phase15_candidate(phase15.artifact_dir)


def test_phase2_spread_bps_fallback_uses_basis_points_units() -> None:
    market_row = phase2_evaluation._market_row_from_example(_shadow_example_without_spread())

    assert market_row.ask_price is not None
    assert market_row.bid_price is not None
    assert market_row.ask_price - market_row.bid_price == pytest.approx(0.05)


def test_phase2_spread_bps_fallback_keeps_execution_costs_bounded() -> None:
    example = _shadow_example_without_spread()
    prediction = PolicyPrediction(
        decision_ts=example.decision_ts,
        source=example.source,
        instrument_id=example.instrument_id,
        action=1.0,
        confidence=1.0,
        regime_embedding=(0.0, 0.0, 5.0, 1_000_000.0),
        score=1.0,
    )

    fills = phase2_evaluation.simulate_execution(
        examples=(example,),
        predictions=(prediction,),
        config=ExecutionSimulationConfig(
            cost_model_config=CostModelConfig(
                fee_bps=0.0,
                base_slippage_bps=0.0,
                volatility_slippage_factor=0.0,
                liquidity_impact_factor=0.0,
            ),
            risk_penalty_factor=0.0,
            policy_edge_scale=0.10,
        ),
    )

    assert len(fills) == 1
    assert fills[0].spread_cost == pytest.approx(0.0005, rel=1e-5)
    assert fills[0].total_execution_cost < 0.001


@pytest.mark.parametrize(
    ("mutate_report", "expected_message"),
    (
        (lambda report: report["metrics"].pop("shadow_sharpe"), "shadow_sharpe"),
        (
            lambda report: report["metrics"]["action_distribution"].pop("mean_abs_turnover"),
            "mean_abs_turnover",
        ),
        (lambda report: report["metrics"].update({"row_count": 0}), "row_count"),
    ),
)
def test_phase2_requires_phase15_shadow_baseline_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate_report,
    expected_message: str,
) -> None:
    phase15 = _accepted_phase15_candidate(tmp_path)
    assert phase15.artifact_dir is not None
    report_path = phase15.artifact_dir / "shadow_acceptance_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    mutate_report(report)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _refresh_artifact_hashes(phase15.artifact_dir)

    def fail_predict(*args, **kwargs):
        raise AssertionError("prediction should not run after baseline metric failure")

    monkeypatch.setattr(XGBoostPolicyModel, "predict_examples", fail_predict)

    with pytest.raises(Phase2ArtifactError, match=expected_message):
        run_phase2_evaluation(
            phase15.artifact_dir,
            phase15.split,
            _phase2_config(),
        )


def test_phase2_costs_reported_do_not_imply_cost_aware_behavior(tmp_path: Path) -> None:
    phase15 = _accepted_phase15_candidate(tmp_path)
    assert phase15.artifact_dir is not None
    candidate = load_phase15_candidate(phase15.artifact_dir)
    baseline_turnover = float(candidate.shadow_baseline_metrics["mean_abs_turnover"])
    fills = (
        ExecutionFill(
            decision_ts=phase15.split.shadow_examples[0].decision_ts,
            source="polymarket",
            instrument_id="btc-updown-15m:UP",
            raw_action=0.5,
            adjusted_action=0.5,
            fill_probability=1.0,
            filled_action=0.5,
            confidence=0.8,
            score=0.8,
            shadow_net_return=0.02,
            gross_policy_return=0.01,
            spread_cost=0.001,
            fee_cost=0.0,
            slippage_cost=0.0,
            liquidity_impact_cost=0.0,
            total_execution_cost=0.001,
            risk_penalty=0.0,
            turnover_penalty=0.0,
            net_execution_return=0.009,
            turnover=baseline_turnover,
            estimated_policy_edge=0.01,
            estimated_friction=0.001,
            expected_net_edge=0.009,
            low_ev_filtered=False,
        ),
        ExecutionFill(
            decision_ts=phase15.split.shadow_examples[1].decision_ts,
            source="polymarket",
            instrument_id="btc-updown-15m:UP",
            raw_action=0.5,
            adjusted_action=0.5,
            fill_probability=1.0,
            filled_action=0.5,
            confidence=0.8,
            score=0.8,
            shadow_net_return=0.018,
            gross_policy_return=0.009,
            spread_cost=0.001,
            fee_cost=0.0,
            slippage_cost=0.0,
            liquidity_impact_cost=0.0,
            total_execution_cost=0.001,
            risk_penalty=0.0,
            turnover_penalty=0.0,
            net_execution_return=0.008,
            turnover=baseline_turnover,
            estimated_policy_edge=0.01,
            estimated_friction=0.001,
            expected_net_edge=0.009,
            low_ev_filtered=False,
        ),
    )

    report = phase2_evaluation.build_phase2_report(
        candidate=candidate,
        fills=fills,
        config=Phase2EvaluationConfig(
            min_sharpe_improvement_ratio=-10.0,
            min_turnover_reduction_ratio=-10.0,
            created_at="2026-06-21T00:10:00Z",
        ),
    )

    assert report.execution_metrics["mean_execution_cost"] > 0.0
    assert report.acceptance_criteria["execution_costs_reported"] is True
    assert report.comparison_metrics["turnover_reduction_ratio"] == pytest.approx(0.0)
    assert report.acceptance_criteria["cost_aware_behavior_emerged"] is False

    bounded_cost_report = phase2_evaluation.build_phase2_report(
        candidate=candidate,
        fills=fills,
        config=Phase2EvaluationConfig(
            min_sharpe_improvement_ratio=-10.0,
            min_turnover_reduction_ratio=-10.0,
            max_cost_to_abs_gross_return_ratio=0.20,
            created_at="2026-06-21T00:10:00Z",
        ),
    )
    assert bounded_cost_report.acceptance_criteria["cost_aware_behavior_emerged"] is True

    strict_behavior_report = phase2_evaluation.build_phase2_report(
        candidate=candidate,
        fills=fills,
        config=Phase2EvaluationConfig(
            min_sharpe_improvement_ratio=-10.0,
            min_turnover_reduction_ratio=-10.0,
            max_cost_to_abs_gross_return_ratio=0.20,
            require_cost_aware_filter_or_turnover_reduction=True,
            created_at="2026-06-21T00:10:00Z",
        ),
    )
    assert strict_behavior_report.acceptance_criteria["cost_aware_behavior_emerged"] is False


def test_phase2_evaluation_does_not_train_and_writes_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase15 = _accepted_phase15_candidate(tmp_path)
    assert phase15.artifact_dir is not None

    def fail_training(*args, **kwargs):
        raise AssertionError("Phase 2 must not train or rerun Phase 1.5")

    monkeypatch.setattr(phase1_training, "run_policy_training", fail_training)
    monkeypatch.setattr(phase1_model, "train_xgboost_policy", fail_training)

    result = run_phase2_evaluation(
        phase15.artifact_dir,
        phase15.split,
        _phase2_config(output_dir=tmp_path / "phase2"),
    )

    assert result.report.phase == PHASE2_EVALUATION_PHASE
    assert result.report.candidate_run_id == phase15.run_manifest["run_id"]
    assert result.report.phase1_5_hashes["model_sha256"] == (
        phase15.run_manifest["artifacts"]["model_sha256"]
    )
    assert result.report.phase1_5_hashes["split_hash"] == phase15.split.split_hash
    assert result.report.phase1_5_shadow_acceptance_metrics["shadow_sharpe"] == (
        phase15.acceptance_report.metrics["shadow_sharpe"]
    )
    assert result.report.execution_metrics["row_count"] == len(phase15.split.shadow_examples)
    assert "mean_net_execution_return" in result.report.execution_metrics
    assert "phase2_execution_sharpe" in result.report.comparison_metrics
    assert result.report.acceptance_criteria["phase1_5_candidate_verified"]
    assert result.report_path is not None
    saved = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert saved["candidate_run_id"] == phase15.run_manifest["run_id"]


def test_phase2_split_mismatch_fails_before_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase15 = _accepted_phase15_candidate(tmp_path)
    assert phase15.artifact_dir is not None
    bad_split = PolicyTrainShadowSplit(
        train_examples=phase15.split.train_examples,
        shadow_examples=phase15.split.shadow_examples,
        split_ts=phase15.split.split_ts,
        split_hash="wrong-split-hash",
        train_dataset_hash=phase15.split.train_dataset_hash,
        shadow_dataset_hash=phase15.split.shadow_dataset_hash,
    )

    def fail_predict(*args, **kwargs):
        raise AssertionError("prediction should not run after split provenance failure")

    monkeypatch.setattr(XGBoostPolicyModel, "predict_examples", fail_predict)

    with pytest.raises(Phase2ArtifactError, match="split_hash"):
        run_phase2_evaluation(
            phase15.artifact_dir,
            bad_split,
            _phase2_config(),
        )
