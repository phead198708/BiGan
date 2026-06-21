"""Phase 0 hard-gate tests for the v8 data-correctness firewall."""

from __future__ import annotations

import math
import random
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from bigan.v8.phase0 import (
    FEATURE_VECTOR_SCHEMA,
    LABEL_SCHEMA,
    CausalFeatureBuilder,
    CostAwareLabelBuilder,
    CostModelConfig,
    FeatureProvenance,
    FeatureVector,
    IntegrityValidator,
    Label,
    MarketDataLoader,
    Phase0Pipeline,
    Phase0PipelineConfig,
    TimeAlignmentEngine,
    TradingCostModel,
    ValidationConfig,
)

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

    report = IntegrityValidator().validate_all(
        features=[bad_feature],
        labels=[],
        market_data=None,
    )

    assert not report.passed
    codes = {failure.code for failure in report.failures}
    assert "feature_causality" in codes
    assert "cross_timeframe_leakage" in codes
    assert "forbidden_feature_name" in codes

