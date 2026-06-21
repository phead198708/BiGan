"""Phase 0 hard-gate tests for the v8 data-correctness firewall."""

from __future__ import annotations

import math
import random
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from bigan.v8.phase0 import (
    FEATURE_VECTOR_SCHEMA,
    LABEL_SCHEMA,
    MARKET_DATA_SCHEMA,
    CausalFeatureBuilder,
    CostAwareLabelBuilder,
    CostCalibrationBucketConfig,
    CostCalibrationConfig,
    CostModelConfig,
    DatasetContract,
    ExecutionCostSample,
    FeatureProvenance,
    FeatureVector,
    IntegrityValidator,
    Label,
    MarketDataLoader,
    Phase0ArtifactError,
    Phase0ArtifactGate,
    Phase0Pipeline,
    Phase0PipelineConfig,
    TimeAlignmentEngine,
    TradingCostModel,
    ValidationConfig,
    assert_phase0_artifact_ready,
)
from bigan.v8.phase0.contracts import schema_names_hash

MINUTE_MS = 60_000


def _ts_at(year: int, month: int, day: int, hour: int, minute: int) -> int:
    return int(datetime(year, month, day, hour, minute, tzinfo=UTC).timestamp() * 1000)


def _market_rows(count: int = 90) -> list[dict]:
    t0 = _ts_at(2026, 6, 1, 12, 0)
    rows = []
    for idx in range(count):
        ts = t0 + idx * MINUTE_MS
        price = 100.0 + 0.22 * math.sin(idx * 0.41) + 0.15 * math.cos(idx * 0.17)
        spread = 0.02 + 0.002 * (idx % 4)
        rows.append(
            {
                "ts": ts,
                "available_at_ts": ts,
                "source": "polymarket",
                "instrument_id": "btc-updown-15m:UP",
                "bid_price": price - spread / 2.0,
                "ask_price": price + spread / 2.0,
                "volume": 10.0 + idx % 7,
                "trade_count": 1 + idx % 3,
                "bid_size": 80.0 + idx % 11,
                "ask_size": 75.0 + (idx * 2) % 13,
                "liquidity_depth": 180.0 + idx % 17,
            }
        )
    return rows


def _pipeline(*, fail_on_validation_error: bool = True) -> Phase0Pipeline:
    return Phase0Pipeline(
        Phase0PipelineConfig(
            cost_config=CostModelConfig(
                fee_bps=5.0,
                base_slippage_bps=1.0,
                volatility_slippage_factor=0.10,
                liquidity_impact_factor=0.002,
            ),
            validation_config=ValidationConfig(
                max_abs_feature_future_corr=0.995,
                min_correlation_rows=25,
            ),
            fail_on_validation_error=fail_on_validation_error,
        )
    )


def _calibration_samples(
    *,
    market_count: int = 20,
    observed_multiplier: float = 1.02,
    cost_model: TradingCostModel | None = None,
) -> tuple[TradingCostModel, list[ExecutionCostSample]]:
    loader = MarketDataLoader()
    market_data = loader.load_rows(_market_rows(market_count))
    resolved_model = cost_model or TradingCostModel(
        CostModelConfig(
            fee_bps=5.0,
            base_slippage_bps=1.0,
            volatility_slippage_factor=0.10,
            liquidity_impact_factor=0.002,
        )
    )
    samples: list[ExecutionCostSample] = []
    for idx, (entry, exit_row) in enumerate(zip(market_data, market_data[1:], strict=False)):
        volatility = 0.001 if idx < market_count // 2 else 0.02
        order_size = 1.0 if idx < market_count // 2 else 25.0
        estimate = resolved_model.estimate(
            entry=entry,
            exit=exit_row,
            order_size=order_size,
            volatility=volatility,
        )
        samples.append(
            ExecutionCostSample(
                entry=entry,
                exit=exit_row,
                order_size=order_size,
                volatility=volatility,
                observed_total_cost=estimate.total_cost * observed_multiplier,
            )
        )
    return resolved_model, samples


def test_phase0_pipeline_is_cost_aware_causal_and_reproducible(tmp_path: Path) -> None:
    rows = _market_rows()
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first = _pipeline().build(list(reversed(rows)), output_dir=first_dir)
    second = _pipeline().build(rows, output_dir=second_dir)

    assert first.validation_report.passed
    assert second.validation_report.passed
    assert first.manifest["dataset_hash"] == second.manifest["dataset_hash"]
    assert first.manifest["feature_rows"] == len(rows)
    assert first.manifest["label_rows"] > 0
    assert first.manifest["validation"]["acceptance_criteria"]["zero_detectable_leakage"]
    assert first.manifest["validation"]["acceptance_criteria"]["statistical_validity_verified"]

    contract = DatasetContract(**first.manifest["dataset_contract"])
    assert contract.dataset_hash == first.manifest["dataset_hash"]
    assert "total_cost" in contract.cost_columns
    assert "net_return" in contract.cost_columns

    assert all(feature.max_input_ts <= feature.decision_ts for feature in first.features)
    assert all(
        provenance.input_end_ts <= feature.decision_ts
        and provenance.available_at_ts <= feature.decision_ts
        for feature in first.features
        for provenance in feature.provenance.values()
    )

    label = first.labels[0]
    assert label.total_cost == pytest.approx(
        label.spread_cost
        + label.fee_cost
        + label.slippage_cost
        + label.liquidity_impact_cost
    )
    assert label.net_return == pytest.approx(label.gross_return - label.total_cost)
    assert label.total_cost > 0.0

    feature_table = pq.read_table(first_dir / "features.parquet")
    label_table = pq.read_table(first_dir / "labels.parquet")
    assert feature_table.schema.names == FEATURE_VECTOR_SCHEMA.names
    assert label_table.schema.names == LABEL_SCHEMA.names


def test_leakage_detection_flags_feature_future_return_correlation() -> None:
    dataset = _pipeline(fail_on_validation_error=False).build(_market_rows())
    shortest_horizon = min(label.horizon_ms for label in dataset.labels)
    label_by_key = {
        (label.source, label.instrument_id, label.decision_ts): label
        for label in dataset.labels
        if label.horizon_ms == shortest_horizon
    }
    leaked_features: list[FeatureVector] = []
    for feature in dataset.features:
        label = label_by_key.get((feature.source, feature.instrument_id, feature.decision_ts))
        if label is None:
            continue
        provenance = feature.provenance["mid_price"].model_copy(
            update={"feature_name": "oracle_signal"}
        )
        leaked_features.append(
            feature.model_copy(
                update={
                    "features": {**feature.features, "oracle_signal": label.net_return},
                    "provenance": {**feature.provenance, "oracle_signal": provenance},
                }
            )
        )

    report = IntegrityValidator(
        ValidationConfig(max_abs_feature_future_corr=0.95, min_correlation_rows=10)
    ).check_feature_future_correlations(
        leaked_features,
        [label for label in dataset.labels if label.horizon_ms == shortest_horizon],
    )

    assert not report.passed
    assert any(
        failure.code == "feature_future_correlation" and failure.column == "oracle_signal"
        for failure in report.failures
    )


def test_time_reversal_shuffle_timestamps_collapses_signal_diagnostic() -> None:
    rng = random.Random(7)
    t0 = _ts_at(2026, 6, 1, 12, 0)
    features: list[FeatureVector] = []
    labels: list[Label] = []
    for idx in range(80):
        decision_ts = t0 + idx * MINUTE_MS
        signal = rng.uniform(-1.0, 1.0)
        provenance = FeatureProvenance(
            feature_name="causal_signal",
            input_start_ts=decision_ts,
            input_end_ts=decision_ts,
            available_at_ts=decision_ts,
            lookback_ms=0,
        )
        features.append(
            FeatureVector(
                decision_ts=decision_ts,
                feature_cutoff_ts=decision_ts,
                lookback_start_ts=decision_ts,
                max_input_ts=decision_ts,
                source="polymarket",
                instrument_id="btc-updown-15m:UP",
                features={"causal_signal": signal},
                provenance={"causal_signal": provenance},
            )
        )
        gross_return = signal / 100.0
        labels.append(
            Label(
                decision_ts=decision_ts,
                label_ts=decision_ts + MINUTE_MS,
                horizon_ms=MINUTE_MS,
                source="polymarket",
                instrument_id="btc-updown-15m:UP",
                entry_price=100.0,
                exit_price=100.0 * (1.0 + gross_return),
                gross_return=gross_return,
                spread_cost=0.0,
                fee_cost=0.0,
                slippage_cost=0.0,
                liquidity_impact_cost=0.0,
                total_cost=0.0,
                net_return=gross_return,
                is_positive=gross_return > 0.0,
            )
        )

    diagnostic = IntegrityValidator().time_reversal_performance_collapse(
        features=features,
        labels=labels,
        feature_column="causal_signal",
        min_corr_drop=0.25,
    )

    assert diagnostic["passed"] is True
    assert diagnostic["baseline_corr"] == pytest.approx(1.0)
    assert abs(float(diagnostic["reversed_corr"])) < 0.35


def test_cost_stress_slippage_multipliers_reduce_net_returns() -> None:
    loader = MarketDataLoader()
    market_data = loader.load_rows(_market_rows())
    aligned = TimeAlignmentEngine().align_market_data(market_data)
    features = CausalFeatureBuilder().build(aligned)
    cost_model = TradingCostModel(
        CostModelConfig(
            fee_bps=5.0,
            base_slippage_bps=1.0,
            volatility_slippage_factor=0.25,
            liquidity_impact_factor=0.002,
        )
    )
    label_builder = CostAwareLabelBuilder(cost_model)
    base = label_builder.build(aligned, features, horizons_ms=(MINUTE_MS,))
    stressed = {
        multiplier: label_builder.build(
            aligned,
            features,
            horizons_ms=(MINUTE_MS,),
            slippage_multiplier=multiplier,
        )
        for multiplier in (1.2, 1.5, 2.0)
    }

    base_by_key = {
        (label.source, label.instrument_id, label.decision_ts, label.horizon_ms): label
        for label in base
    }
    for multiplier, labels in stressed.items():
        for label in labels:
            key = (label.source, label.instrument_id, label.decision_ts, label.horizon_ms)
            base_label = base_by_key[key]
            assert label.slippage_cost == pytest.approx(
                base_label.slippage_cost * multiplier
            )
            assert label.total_cost > base_label.total_cost
            assert label.net_return < base_label.net_return


def test_label_consistency_across_horizons_and_market_prices() -> None:
    rows = _market_rows()
    dataset = _pipeline(fail_on_validation_error=False).build(rows)
    validator = IntegrityValidator()

    report = validator.validate_label_consistency(dataset.labels, market_data=dataset.market_data)

    assert report.passed
    by_decision: dict[tuple[str, str, int], list[Label]] = {}
    for label in dataset.labels:
        by_decision.setdefault(
            (label.source, label.instrument_id, label.decision_ts),
            [],
        ).append(label)
    for group in by_decision.values():
        ordered = sorted(group, key=lambda label: label.horizon_ms)
        assert [label.label_ts for label in ordered] == sorted(label.label_ts for label in ordered)

    tampered = dataset.labels[0].model_copy(
        update={
            "gross_return": dataset.labels[0].gross_return + 0.10,
            "net_return": dataset.labels[0].net_return + 0.10,
            "is_positive": dataset.labels[0].net_return + 0.10 > 0.0,
        }
    )
    bad_report = validator.validate_label_consistency(
        [tampered, *dataset.labels[1:]],
        market_data=dataset.market_data,
    )

    assert not bad_report.passed
    assert any(failure.code == "label_market_consistency" for failure in bad_report.failures)


def test_timestamp_and_cross_timeframe_leakage_are_rejected() -> None:
    dataset = _pipeline(fail_on_validation_error=False).build(_market_rows())
    feature = dataset.features[10]
    future_provenance = FeatureProvenance(
        feature_name="cross_timeframe_signal",
        input_start_ts=feature.decision_ts,
        input_end_ts=feature.decision_ts + MINUTE_MS,
        available_at_ts=feature.decision_ts + MINUTE_MS,
        lookback_ms=15 * MINUTE_MS,
        source_timeframe_ms=15 * MINUTE_MS,
    )
    forbidden_provenance = feature.provenance["mid_price"].model_copy(
        update={"feature_name": "future_return_1m"}
    )
    bad_feature = feature.model_copy(
        update={
            "features": {
                **feature.features,
                "cross_timeframe_signal": 1.0,
                "future_return_1m": 0.1,
            },
            "provenance": {
                **feature.provenance,
                "cross_timeframe_signal": future_provenance,
                "future_return_1m": forbidden_provenance,
            },
        }
    )

    matching_label = next(
        label
        for label in dataset.labels
        if label.source == bad_feature.source
        and label.instrument_id == bad_feature.instrument_id
        and label.decision_ts == bad_feature.decision_ts
    )
    report = IntegrityValidator().validate_all(
        features=[bad_feature],
        labels=[matching_label],
        market_data=None,
    )

    assert not report.passed
    codes = {failure.code for failure in report.failures}
    assert "feature_causality" in codes
    assert "feature_label_causality" in codes
    assert "cross_timeframe_leakage" in codes
    assert "forbidden_feature_name" in codes


def test_statistical_integrity_detects_distribution_drift() -> None:
    t0 = _ts_at(2026, 6, 1, 12, 0)
    features: list[FeatureVector] = []
    for idx in range(100):
        decision_ts = t0 + idx * MINUTE_MS
        value = 0.0 if idx < 70 else 10.0
        provenance = FeatureProvenance(
            feature_name="distribution_probe",
            input_start_ts=decision_ts,
            input_end_ts=decision_ts,
            available_at_ts=decision_ts,
            lookback_ms=0,
        )
        features.append(
            FeatureVector(
                decision_ts=decision_ts,
                feature_cutoff_ts=decision_ts,
                lookback_start_ts=decision_ts,
                max_input_ts=decision_ts,
                source="polymarket",
                instrument_id="btc-updown-15m:UP",
                features={"distribution_probe": value},
                provenance={"distribution_probe": provenance},
            )
        )

    report = IntegrityValidator(
        ValidationConfig(
            min_drift_rows=20,
            max_ks_statistic=0.2,
            max_psi=0.2,
            max_kl_divergence=0.2,
        )
    ).validate_statistical_integrity(features)

    assert not report.passed
    assert any(
        failure.code == "feature_distribution_drift"
        and failure.column == "distribution_probe"
        for failure in report.failures
    )
    metrics = report.metrics["statistical_integrity"]["metrics"]["distribution_probe"]
    assert metrics["ks_statistic"] == pytest.approx(1.0)
    assert metrics["psi"] > 0.2
    assert metrics["kl_divergence"] > 0.2


def test_cost_model_calibration_validates_against_observed_execution_costs() -> None:
    cost_model, samples = _calibration_samples(market_count=10)

    calibration_config = CostCalibrationConfig(
        min_samples=5,
        max_mean_absolute_error=0.001,
        max_mean_absolute_percentage_error=0.05,
        max_abs_bias=0.001,
        max_weighted_mean_absolute_percentage_error=0.05,
        max_median_absolute_percentage_error=0.05,
    )
    report = cost_model.validate_calibration(samples, config=calibration_config)

    assert report.passed
    assert report.sample_count == len(samples)
    assert report.mean_absolute_percentage_error == pytest.approx(0.02 / 1.02, rel=1e-2)
    assert report.weighted_mean_absolute_percentage_error == pytest.approx(
        0.02 / 1.02,
        rel=1e-2,
    )
    assert report.median_absolute_error is not None
    assert report.median_absolute_percentage_error is not None
    assert report.symmetric_mean_absolute_percentage_error is not None

    bad_samples = [
        ExecutionCostSample(
            entry=sample.entry,
            exit=sample.exit,
            order_size=sample.order_size,
            volatility=sample.volatility,
            observed_total_cost=sample.observed_total_cost * 4.0,
        )
        for sample in samples
    ]
    bad_report = cost_model.validate_calibration(bad_samples, config=calibration_config)

    assert not bad_report.passed
    assert bad_report.mean_absolute_percentage_error is not None
    assert bad_report.mean_absolute_percentage_error > 0.05


def test_artifact_gate_accepts_valid_manifest_and_rejects_invalid_contracts() -> None:
    dataset = _pipeline().build(_market_rows())
    contract = assert_phase0_artifact_ready(dataset.manifest)
    assert contract.dataset_hash == dataset.manifest["dataset_hash"]

    invalid_cases = [
        ("missing_dataset_contract", lambda manifest: manifest.pop("dataset_contract")),
        ("dataset_hash_mismatch", lambda manifest: manifest.update({"dataset_hash": "stale"})),
        (
            "stale_dataset_version",
            lambda manifest: manifest["dataset_contract"].update(
                {"dataset_version": "bigan-v8-phase0-v0"}
            ),
        ),
        (
            "missing_cost_columns",
            lambda manifest: manifest["dataset_contract"].update(
                {
                    "cost_columns": [
                        column
                        for column in manifest["dataset_contract"]["cost_columns"]
                        if column != "net_return"
                    ]
                }
            ),
        ),
        (
            "validation_failed",
            lambda manifest: manifest["validation"].update({"passed": False}),
        ),
        (
            "missing_acceptance_criteria",
            lambda manifest: manifest["validation"].pop("acceptance_criteria"),
        ),
        (
            "acceptance_criterion_failed",
            lambda manifest: manifest["validation"]["acceptance_criteria"].update(
                {"zero_detectable_leakage": False}
            ),
        ),
    ]
    for expected_code, mutate in invalid_cases:
        manifest = deepcopy(dataset.manifest)
        mutate(manifest)
        report = Phase0ArtifactGate().validate_manifest(manifest)
        assert not report.passed
        assert expected_code in {failure.code for failure in report.failures}
        with pytest.raises(Phase0ArtifactError):
            report.raise_if_failed()


def test_artifact_gate_separates_required_schema_and_canonical_order() -> None:
    dataset = _pipeline().build(_market_rows())
    reordered_manifest = deepcopy(dataset.manifest)
    reordered_manifest["dataset_contract"]["feature_schema"] = list(
        reversed(FEATURE_VECTOR_SCHEMA.names)
    )
    reordered_manifest["dataset_contract"]["feature_schema_hash"] = schema_names_hash(
        tuple(reordered_manifest["dataset_contract"]["feature_schema"])
    )

    strict_report = Phase0ArtifactGate().validate_manifest(reordered_manifest)
    assert not strict_report.passed
    assert "feature_schema_order_mismatch" in {
        failure.code for failure in strict_report.failures
    }

    relaxed_report = Phase0ArtifactGate(require_canonical_order=False).validate_manifest(
        reordered_manifest
    )
    assert relaxed_report.passed

    missing_manifest = deepcopy(dataset.manifest)
    missing_manifest["dataset_contract"]["market_schema"] = [
        column
        for column in MARKET_DATA_SCHEMA.names
        if column != "available_at_ts"
    ]
    missing_report = Phase0ArtifactGate(require_canonical_order=False).validate_manifest(
        missing_manifest
    )
    assert not missing_report.passed
    assert "market_schema_missing_columns" in {
        failure.code for failure in missing_report.failures
    }


def test_artifact_gate_rejects_schema_hash_and_manifest_version_mismatches() -> None:
    dataset = _pipeline().build(_market_rows())
    hash_cases = [
        ("market_schema_hash", "market_schema_hash_mismatch"),
        ("feature_schema_hash", "feature_schema_hash_mismatch"),
        ("label_schema_hash", "label_schema_hash_mismatch"),
    ]
    for hash_field, expected_code in hash_cases:
        manifest = deepcopy(dataset.manifest)
        manifest["dataset_contract"][hash_field] = "stale"
        report = Phase0ArtifactGate(require_canonical_order=False).validate_manifest(manifest)
        assert not report.passed
        assert expected_code in {failure.code for failure in report.failures}

    missing_version = deepcopy(dataset.manifest)
    missing_version.pop("dataset_version")
    missing_report = Phase0ArtifactGate().validate_manifest(missing_version)
    assert "missing_manifest_dataset_version" in {
        failure.code for failure in missing_report.failures
    }

    mismatch_version = deepcopy(dataset.manifest)
    mismatch_version["dataset_version"] = "bigan-v8-phase0-v0"
    mismatch_report = Phase0ArtifactGate().validate_manifest(mismatch_version)
    assert "manifest_dataset_version_mismatch" in {
        failure.code for failure in mismatch_report.failures
    }


def test_pipeline_manifest_cost_calibration_is_gate_enforced() -> None:
    cost_model, samples = _calibration_samples(market_count=24)
    pipeline = Phase0Pipeline(
        Phase0PipelineConfig(
            cost_config=cost_model.config,
            require_cost_calibration=True,
            cost_calibration_config=CostCalibrationConfig(
                min_samples=5,
                max_mean_absolute_error=0.001,
                max_mean_absolute_percentage_error=0.05,
                max_abs_bias=0.001,
            ),
            cost_calibration_bucket_config=CostCalibrationBucketConfig(
                min_bucket_samples=5,
                bucket_by_source=False,
                bucket_by_instrument=False,
                volatility_edges=(0.005,),
                spread_edges=(),
                liquidity_edges=(),
                order_size_edges=(5.0,),
            ),
        )
    )

    dataset = pipeline.build(_market_rows(), cost_calibration_samples=samples)

    assert dataset.manifest["cost_calibration"]["passed"] is True
    assert (
        dataset.manifest["validation"]["acceptance_criteria"]["cost_model_realistic"]
        is True
    )
    assert Phase0ArtifactGate(require_cost_calibration=True).validate_manifest(
        dataset.manifest
    ).passed

    missing_calibration = deepcopy(dataset.manifest)
    missing_calibration.pop("cost_calibration")
    missing_report = Phase0ArtifactGate(require_cost_calibration=True).validate_manifest(
        missing_calibration
    )
    assert "missing_cost_calibration" in {
        failure.code for failure in missing_report.failures
    }

    failed_calibration = deepcopy(dataset.manifest)
    failed_calibration["cost_calibration"]["passed"] = False
    failed_report = Phase0ArtifactGate(require_cost_calibration=True).validate_manifest(
        failed_calibration
    )
    assert "cost_calibration_failed" in {
        failure.code for failure in failed_report.failures
    }

    failed_bucket = deepcopy(dataset.manifest)
    failed_bucket["cost_calibration"]["failed_buckets"] = ["volatility=>=0.005"]
    bucket_report = Phase0ArtifactGate(require_cost_calibration=True).validate_manifest(
        failed_bucket
    )
    assert "cost_calibration_bucket_failed" in {
        failure.code for failure in bucket_report.failures
    }

    bad_samples = [
        ExecutionCostSample(
            entry=sample.entry,
            exit=sample.exit,
            order_size=sample.order_size,
            volatility=sample.volatility,
            observed_total_cost=sample.observed_total_cost * 4.0,
        )
        for sample in samples
    ]
    bad_pipeline = Phase0Pipeline(
        Phase0PipelineConfig(
            cost_config=cost_model.config,
            require_cost_calibration=True,
            fail_on_validation_error=False,
        )
    )
    bad_dataset = bad_pipeline.build(_market_rows(), cost_calibration_samples=bad_samples)
    assert (
        bad_dataset.manifest["validation"]["acceptance_criteria"]["cost_model_realistic"]
        is False
    )


def test_bucketed_cost_calibration_fails_when_sampled_regime_fails() -> None:
    cost_model, samples = _calibration_samples(market_count=24)
    calibration_config = CostCalibrationConfig(
        min_samples=5,
        max_mean_absolute_error=0.001,
        max_mean_absolute_percentage_error=0.05,
        max_abs_bias=0.001,
    )
    bucket_config = CostCalibrationBucketConfig(
        min_bucket_samples=5,
        bucket_by_source=False,
        bucket_by_instrument=False,
        volatility_edges=(0.005,),
        spread_edges=(),
        liquidity_edges=(),
        order_size_edges=(5.0,),
    )

    good_report = cost_model.validate_calibration_by_bucket(
        samples,
        bucket_config=bucket_config,
        config=calibration_config,
    )
    assert good_report.passed
    assert good_report.buckets
    assert not good_report.failed_buckets

    bad_samples = [
        sample
        if (sample.volatility or 0.0) < 0.005
        else ExecutionCostSample(
            entry=sample.entry,
            exit=sample.exit,
            order_size=sample.order_size,
            volatility=sample.volatility,
            observed_total_cost=sample.observed_total_cost * 3.0,
        )
        for sample in samples
    ]
    bad_report = cost_model.validate_calibration_by_bucket(
        bad_samples,
        bucket_config=bucket_config,
        config=calibration_config,
    )
    assert not bad_report.passed
    assert bad_report.aggregate.passed is False
    assert bad_report.failed_buckets


def test_sparse_bucket_calibration_cannot_silently_pass() -> None:
    cost_model, samples = _calibration_samples(market_count=24)
    calibration_config = CostCalibrationConfig(
        min_samples=5,
        max_mean_absolute_error=0.001,
        max_mean_absolute_percentage_error=0.05,
        max_abs_bias=0.001,
    )

    all_skipped = cost_model.validate_calibration_by_bucket(
        samples,
        config=calibration_config,
        bucket_config=CostCalibrationBucketConfig(
            min_bucket_samples=100,
            bucket_by_source=False,
            bucket_by_instrument=False,
            volatility_edges=(0.005,),
            spread_edges=(),
            liquidity_edges=(),
            order_size_edges=(),
        ),
    )
    assert not all_skipped.passed
    assert all_skipped.checked_bucket_count == 0
    assert all_skipped.skipped_sample_count == len(samples)

    ratio_failed = cost_model.validate_calibration_by_bucket(
        samples,
        config=calibration_config,
        bucket_config=CostCalibrationBucketConfig(
            min_bucket_samples=12,
            min_checked_sample_ratio=0.80,
            bucket_by_source=False,
            bucket_by_instrument=False,
            volatility_edges=(0.005,),
            spread_edges=(),
            liquidity_edges=(),
            order_size_edges=(5.0,),
        ),
    )
    assert not ratio_failed.passed
    assert ratio_failed.checked_sample_ratio < 0.80

    bucket_count_failed = cost_model.validate_calibration_by_bucket(
        samples,
        config=calibration_config,
        bucket_config=CostCalibrationBucketConfig(
            min_bucket_samples=5,
            min_checked_bucket_count=3,
            bucket_by_source=False,
            bucket_by_instrument=False,
            volatility_edges=(0.005,),
            spread_edges=(),
            liquidity_edges=(),
            order_size_edges=(5.0,),
        ),
    )
    assert not bucket_count_failed.passed
    assert bucket_count_failed.checked_bucket_count < 3


def test_robust_low_cost_metrics_remain_finite_for_tiny_observed_costs() -> None:
    loader = MarketDataLoader()
    market_data = loader.load_rows(_market_rows(8))
    cost_model = TradingCostModel(CostModelConfig(liquidity_impact_factor=0.0))
    samples = [
        ExecutionCostSample(
            entry=entry,
            exit=exit_row,
            observed_total_cost=1e-9,
            volatility=0.0,
        )
        for entry, exit_row in zip(market_data, market_data[1:], strict=False)
    ]

    report = cost_model.validate_calibration(
        samples,
        config=CostCalibrationConfig(
            min_samples=5,
            percentage_error_floor=1e-4,
            max_mean_absolute_error=1.0,
            max_mean_absolute_percentage_error=1e6,
            max_abs_bias=1.0,
        ),
    )

    assert report.mean_absolute_percentage_error is not None
    assert math.isfinite(report.mean_absolute_percentage_error)
    assert report.weighted_mean_absolute_percentage_error is not None
    assert math.isfinite(report.weighted_mean_absolute_percentage_error)
    assert report.median_absolute_percentage_error is not None
    assert math.isfinite(report.median_absolute_percentage_error)


def test_drift_policy_supports_warning_and_insufficient_row_hard_fail() -> None:
    t0 = _ts_at(2026, 6, 1, 12, 0)
    features = [
        FeatureVector(
            decision_ts=t0 + idx * MINUTE_MS,
            feature_cutoff_ts=t0 + idx * MINUTE_MS,
            lookback_start_ts=t0 + idx * MINUTE_MS,
            max_input_ts=t0 + idx * MINUTE_MS,
            source="polymarket",
            instrument_id="btc-updown-15m:UP",
            features={"probe": float(idx)},
            provenance={
                "probe": FeatureProvenance(
                    feature_name="probe",
                    input_start_ts=t0 + idx * MINUTE_MS,
                    input_end_ts=t0 + idx * MINUTE_MS,
                    available_at_ts=t0 + idx * MINUTE_MS,
                    lookback_ms=0,
                )
            },
        )
        for idx in range(10)
    ]

    hard_report = IntegrityValidator(
        ValidationConfig(
            min_drift_rows=20,
            fail_on_insufficient_drift_rows=True,
        )
    ).validate_statistical_integrity(features)
    assert not hard_report.passed
    assert "feature_distribution_insufficient_rows" in {
        failure.code for failure in hard_report.failures
    }

    warn_report = IntegrityValidator(
        ValidationConfig(
            min_drift_rows=20,
            fail_on_insufficient_drift_rows=True,
            statistical_integrity_mode="warn",
        )
    ).validate_statistical_integrity(features)
    assert warn_report.passed
    assert warn_report.failures[0].severity == "warning"
