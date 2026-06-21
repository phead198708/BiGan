"""Phase 1 hard-gate tests for pure policy learning."""

from __future__ import annotations

import inspect
import math
from copy import deepcopy
from datetime import UTC, datetime

import pytest

import bigan.v8.phase1.model as phase1_model
from bigan.v8.phase0 import (
    CostModelConfig,
    FeatureProvenance,
    FeatureVector,
    Label,
    Phase0ArtifactError,
    Phase0Pipeline,
    Phase0PipelineConfig,
    ValidationConfig,
)
from bigan.v8.phase1 import (
    PolicyAcceptanceConfig,
    PolicyDatasetConfig,
    PolicyPrediction,
    PolicyTrainingExample,
    PolicyTrainShadowSplit,
    XGBoostPolicyConfig,
    build_policy_dataset,
    build_policy_dataset_from_phase0,
    build_temporal_policy_split,
    policy_dataset_hash,
    train_xgboost_policy,
    validate_policy_acceptance,
    validate_policy_shadow_split,
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


def _causal_policy_rows(row_count: int = 80) -> tuple[list[FeatureVector], list[Label]]:
    t0 = _ts_at(2026, 6, 1, 12, 0)
    features: list[FeatureVector] = []
    labels: list[Label] = []
    for idx in range(row_count):
        decision_ts = t0 + idx * MINUTE_MS
        signal = -1.0 if idx % 4 in (0, 1) else 1.0
        shadow_net_return = 0.020 if signal > 0.0 else -0.015
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
                exit_price=100.0 * (1.0 + shadow_net_return),
                gross_return=shadow_net_return,
                spread_cost=0.0,
                fee_cost=0.0,
                slippage_cost=0.0,
                liquidity_impact_cost=0.0,
                total_cost=0.0,
                net_return=shadow_net_return,
                is_positive=shadow_net_return > 0.0,
            )
        )
    return features, labels


def _causal_policy_dataset(
    row_count: int = 80,
    *,
    target_encoding: str = "binary_positive_net_return_threshold",
):
    features, labels = _causal_policy_rows(row_count)
    return build_policy_dataset(
        features=features,
        labels=labels,
        phase0_dataset_hash="phase0-hash-for-policy-test",
        phase0_dataset_version="bigan-v8-phase0-v1.0.0",
        config=PolicyDatasetConfig(
            horizon_ms=MINUTE_MS,
            target_encoding=target_encoding,  # type: ignore[arg-type]
            rank_quality_bucket_edges=(-0.005, 0.0, 0.005),
            feature_columns=(
                "signal",
                "volatility_5m",
                "volatility_15m",
                "spread_bps",
                "liquidity_depth",
            ),
        ),
    )


def _copy_example(
    example: PolicyTrainingExample,
    *,
    target_label: float | None = None,
    shadow_net_return: float | None = None,
    regime_key: str | None = None,
) -> PolicyTrainingExample:
    return PolicyTrainingExample(
        decision_ts=example.decision_ts,
        source=example.source,
        instrument_id=example.instrument_id,
        features=example.features,
        target_label=example.target_label if target_label is None else target_label,
        shadow_net_return=(
            example.shadow_net_return
            if shadow_net_return is None
            else shadow_net_return
        ),
        horizon_ms=example.horizon_ms,
        regime_key=example.regime_key if regime_key is None else regime_key,
    )


def _dataset_with_examples(dataset, examples: tuple[PolicyTrainingExample, ...]):
    return dataset.__class__(
        examples=examples,
        feature_columns=dataset.feature_columns,
        policy_dataset_hash=policy_dataset_hash(
            examples=examples,
            feature_columns=dataset.feature_columns,
            phase0_dataset_hash=dataset.phase0_dataset_hash,
            phase0_dataset_version=dataset.phase0_dataset_version,
            config=dataset.config,
        ),
        phase0_dataset_hash=dataset.phase0_dataset_hash,
        phase0_dataset_version=dataset.phase0_dataset_version,
        config=dataset.config,
    )


def test_phase1_dataset_adapter_requires_ready_phase0_and_is_deterministic() -> None:
    phase0_dataset = _pipeline().build(_market_rows())

    first = build_policy_dataset_from_phase0(phase0_dataset)
    second = build_policy_dataset_from_phase0(phase0_dataset)

    assert first.policy_dataset_hash == second.policy_dataset_hash
    assert first.phase0_dataset_hash == phase0_dataset.manifest["dataset_hash"]
    assert first.phase0_dataset_version == phase0_dataset.manifest["dataset_version"]
    assert first.to_dict()["row_count"] == len(first.examples)
    assert first.config.target_encoding == "binary_positive_net_return_threshold"
    assert "net_return" not in first.feature_columns
    assert "total_cost" not in first.feature_columns
    assert all(example.target_label in {0.0, 1.0} for example in first.examples)
    assert all(math.isfinite(example.shadow_net_return) for example in first.examples)
    assert first.policy_dataset_hash == policy_dataset_hash(
        examples=first.examples,
        feature_columns=first.feature_columns,
        phase0_dataset_hash=first.phase0_dataset_hash,
        phase0_dataset_version=first.phase0_dataset_version,
        config=first.config,
    )

    failed_manifest = deepcopy(phase0_dataset.manifest)
    failed_manifest["validation"]["acceptance_criteria"]["zero_detectable_leakage"] = False
    unsafe_phase0 = phase0_dataset
    unsafe_phase0.manifest = failed_manifest
    with pytest.raises(Phase0ArtifactError, match="zero_detectable_leakage"):
        build_policy_dataset_from_phase0(unsafe_phase0)


def test_phase1_dataset_rejects_label_cost_and_empty_feature_columns() -> None:
    features, labels = _causal_policy_rows(row_count=8)
    with pytest.raises(ValueError, match="label/cost columns"):
        build_policy_dataset(
            features=features,
            labels=labels,
            phase0_dataset_hash="phase0-hash-for-policy-test",
            phase0_dataset_version="bigan-v8-phase0-v1.0.0",
            config=PolicyDatasetConfig(feature_columns=("net_return",)),
        )
    with pytest.raises(ValueError, match="no usable values"):
        build_policy_dataset(
            features=features,
            labels=labels,
            phase0_dataset_hash="phase0-hash-for-policy-test",
            phase0_dataset_version="bigan-v8-phase0-v1.0.0",
            config=PolicyDatasetConfig(feature_columns=("missing_signal",)),
        )


def test_temporal_policy_split_is_strict_deterministic_and_non_overlapping() -> None:
    dataset = _causal_policy_dataset(row_count=30)

    first = build_temporal_policy_split(dataset, train_fraction=0.60)
    second = build_temporal_policy_split(dataset, train_fraction=0.60)

    assert first.split_hash == second.split_hash
    assert first.train_dataset_hash == second.train_dataset_hash
    assert first.shadow_dataset_hash == second.shadow_dataset_hash
    assert max(example.decision_ts for example in first.train_examples) < min(
        example.decision_ts for example in first.shadow_examples
    )
    assert first.to_dict()["train_row_count"] == len(first.train_examples)
    assert first.to_dict()["shadow_row_count"] == len(first.shadow_examples)

    with pytest.raises(ValueError, match="train_examples"):
        build_temporal_policy_split(dataset, split_ts=dataset.examples[0].decision_ts)
    with pytest.raises(ValueError, match="shadow_examples"):
        build_temporal_policy_split(
            dataset,
            split_ts=dataset.examples[-1].decision_ts + MINUTE_MS,
        )
    with pytest.raises(ValueError, match="max\\(train_ts\\) < min\\(shadow_ts\\)"):
        PolicyTrainShadowSplit(
            train_examples=(dataset.examples[2],),
            shadow_examples=(dataset.examples[1],),
            split_ts=dataset.examples[2].decision_ts,
            split_hash="split",
            train_dataset_hash="train",
            shadow_dataset_hash="shadow",
        )


def test_xgboost_policy_rejects_direct_pnl_optimization_knobs_and_bad_actions() -> None:
    with pytest.raises(ValueError, match="PnL/profit"):
        XGBoostPolicyConfig(selection_metric="shadow_sharpe")
    with pytest.raises(ValueError, match="PnL/profit"):
        XGBoostPolicyConfig(eval_metric="profit_factor")
    with pytest.raises(ValueError):
        XGBoostPolicyConfig(objective="pnl:max")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="action must be in \\[0, 1\\]"):
        PolicyPrediction(
            decision_ts=1,
            source="polymarket",
            instrument_id="btc-updown-15m:UP",
            action=1.01,
            confidence=0.5,
            regime_embedding=(0.0,),
            score=0.5,
        )


def test_xgboost_policy_trains_on_train_split_and_shadow_acceptance_uses_shadow_rows() -> None:
    dataset = _causal_policy_dataset()
    split = build_temporal_policy_split(dataset, train_fraction=0.65)
    config = XGBoostPolicyConfig(
        num_boost_round=8,
        max_depth=2,
        learning_rate=0.2,
        action_activation_threshold=0.40,
    )

    model = train_xgboost_policy(dataset, config, split=split)

    assert model.training_manifest["direct_pnl_optimization"] is False
    assert model.training_manifest["shadow_return_used_for_training"] is False
    assert model.training_manifest["training_label_field"] == "target_label"
    assert model.training_manifest["target_encoding"] == dataset.config.target_encoding
    assert model.training_manifest["row_count"] == len(split.train_examples)
    assert model.training_manifest["train_dataset_hash"] == split.train_dataset_hash
    assert model.training_manifest["shadow_dataset_hash"] == split.shadow_dataset_hash
    assert model.training_manifest["split"]["split_hash"] == split.split_hash

    class FakePolicyModel:
        observed_examples: tuple[PolicyTrainingExample, ...] = ()

        def predict_examples(
            self,
            examples: tuple[PolicyTrainingExample, ...],
        ) -> tuple[PolicyPrediction, ...]:
            self.observed_examples = examples
            return tuple(
                PolicyPrediction(
                    decision_ts=example.decision_ts,
                    source=example.source,
                    instrument_id=example.instrument_id,
                    action=0.5 if example.target_label > 0.0 else 0.0,
                    confidence=0.75,
                    regime_embedding=(0.01, 0.02, 2.0, 100.0),
                    score=example.target_label,
                )
                for example in examples
            )

    fake_model = FakePolicyModel()
    fake_model.training_manifest = model.training_manifest
    report = validate_policy_shadow_split(
        fake_model,
        split,
        PolicyAcceptanceConfig(min_active_regime_count=1),
    )

    assert fake_model.observed_examples == split.shadow_examples
    assert report.metrics["evaluation_scope"] == "shadow"
    assert report.metrics["split_hash"] == split.split_hash
    assert report.metrics["training_row_count"] == len(split.train_examples)
    assert report.metrics["row_count"] == len(split.shadow_examples)
    assert report.metrics["split_provenance"]["passed"] is True
    assert report.acceptance_criteria["split_provenance_verified"] is True


def test_shadow_split_validation_rejects_missing_or_mismatched_training_provenance() -> None:
    dataset = _causal_policy_dataset(row_count=40)
    split = build_temporal_policy_split(dataset, train_fraction=0.60)
    other_split = build_temporal_policy_split(dataset, train_fraction=0.75)
    config = XGBoostPolicyConfig(num_boost_round=4, max_depth=2)

    class NoManifestPolicyModel:
        def predict_examples(
            self,
            examples: tuple[PolicyTrainingExample, ...],
        ) -> tuple[PolicyPrediction, ...]:
            raise AssertionError("predict_examples must not run when provenance fails")

    missing_report = validate_policy_shadow_split(
        NoManifestPolicyModel(),
        split,
        PolicyAcceptanceConfig(min_active_regime_count=1),
    )

    assert missing_report.acceptance_criteria["split_provenance_verified"] is False
    assert missing_report.metrics["prediction_skipped_due_to_split_provenance"] is True
    assert missing_report.metrics["split_provenance"]["training_manifest_present"] is False
    assert "missing_training_manifest" in {
        failure.code for failure in missing_report.failures
    }

    full_dataset_model = train_xgboost_policy(dataset, config)
    no_split_report = validate_policy_shadow_split(
        full_dataset_model,
        split,
        PolicyAcceptanceConfig(min_active_regime_count=1),
    )

    assert no_split_report.acceptance_criteria["split_provenance_verified"] is False
    assert no_split_report.metrics["prediction_skipped_due_to_split_provenance"] is True
    assert no_split_report.metrics["split_provenance"]["split_hash_matches"] is False
    assert "split_hash_mismatch" in {
        failure.code for failure in no_split_report.failures
    }

    split_model = train_xgboost_policy(dataset, config, split=split)
    mismatch_report = validate_policy_shadow_split(
        split_model,
        other_split,
        PolicyAcceptanceConfig(min_active_regime_count=1),
    )

    assert mismatch_report.acceptance_criteria["split_provenance_verified"] is False
    assert mismatch_report.metrics["prediction_skipped_due_to_split_provenance"] is True
    failure_codes = {failure.code for failure in mismatch_report.failures}
    assert "split_hash_mismatch" in failure_codes
    assert "train_split_mismatch" in failure_codes
    assert "shadow_split_mismatch" in failure_codes


def test_xgboost_ranking_uses_discrete_target_labels_and_auditable_groups() -> None:
    dataset = _causal_policy_dataset(
        target_encoding="rank_discrete_net_return_quality_bucket"
    )
    config = XGBoostPolicyConfig(
        objective="rank:pairwise",
        eval_metric="ndcg",
        selection_metric="ndcg",
        num_boost_round=4,
        max_depth=2,
        ranking_group_strategy="source_instrument",
    )

    model = train_xgboost_policy(dataset, config)

    assert model.training_manifest["objective_type"] == "pairwise_ranking_policy"
    assert model.training_manifest["target_encoding"] == (
        "rank_discrete_net_return_quality_bucket"
    )
    assert model.training_manifest["training_label_field"] == "target_label"
    assert model.training_manifest["shadow_return_used_for_training"] is False
    assert model.training_manifest["ranking_group_strategy"] == "source_instrument"
    assert model.training_manifest["ranking_group_count"] >= 1
    assert model.training_manifest["ranking_effective_group_count"] >= 1
    assert model.training_manifest["ranking_ineffective_group_count"] >= 0
    assert all(size > 0 for size in model.training_manifest["ranking_group_sizes"])
    assert len(model.training_manifest["ranking_group_sizes"]) == len(
        model.training_manifest["ranking_group_keys"]
    )

    labels_source = inspect.getsource(phase1_model._training_labels)
    assert "target_label" in labels_source
    assert "shadow_net_return" not in labels_source
    assert "net_return" not in labels_source


def test_xgboost_ranking_rejects_ineffective_ranking_groups() -> None:
    dataset = _causal_policy_dataset(
        row_count=12,
        target_encoding="rank_discrete_net_return_quality_bucket",
    )
    config = XGBoostPolicyConfig(
        objective="rank:pairwise",
        eval_metric="ndcg",
        selection_metric="ndcg",
        num_boost_round=2,
        max_depth=2,
        ranking_group_strategy="source_instrument_regime",
    )

    size_one_group_dataset = _dataset_with_examples(
        dataset,
        tuple(
            _copy_example(example, regime_key=f"unique-regime-{idx}")
            for idx, example in enumerate(dataset.examples)
        ),
    )
    with pytest.raises(ValueError, match="effective ranking groups"):
        train_xgboost_policy(size_one_group_dataset, config)

    half = len(dataset.examples) // 2
    constant_label_group_dataset = _dataset_with_examples(
        dataset,
        tuple(
            _copy_example(
                example,
                target_label=0.0 if idx < half else 1.0,
                regime_key="constant-zero" if idx < half else "constant-one",
            )
            for idx, example in enumerate(dataset.examples)
        ),
    )
    with pytest.raises(ValueError, match="effective ranking groups"):
        train_xgboost_policy(constant_label_group_dataset, config)


def test_objective_and_target_encoding_must_match() -> None:
    binary_dataset = _causal_policy_dataset()
    ranking_dataset = _causal_policy_dataset(
        target_encoding="rank_discrete_net_return_quality_bucket"
    )

    with pytest.raises(ValueError, match="requires target_encoding"):
        train_xgboost_policy(
            binary_dataset,
            XGBoostPolicyConfig(
                objective="rank:pairwise",
                eval_metric="ndcg",
                selection_metric="ndcg",
            ),
        )
    with pytest.raises(ValueError, match="requires target_encoding"):
        train_xgboost_policy(ranking_dataset, XGBoostPolicyConfig())


def test_policy_acceptance_validates_shadow_sharpe_distribution_regimes_and_buckets() -> None:
    dataset = _causal_policy_dataset(row_count=12)
    actions = (0.0, 0.0, 0.50, 0.50, 1.0, 1.0, 0.0, 0.50, 1.0, 0.0, 0.50, 1.0)
    examples = tuple(
        _copy_example(
            example,
            shadow_net_return=(
                -0.010
                if action == 0.0
                else 0.015
                if action == 0.50
                else 0.030
            ),
            regime_key="regime-a" if idx % 2 == 0 else "regime-b",
        )
        for idx, (example, action) in enumerate(zip(dataset.examples, actions, strict=True))
    )
    predictions = tuple(
        PolicyPrediction(
            decision_ts=example.decision_ts,
            source=example.source,
            instrument_id=example.instrument_id,
            action=action,
            confidence=0.75,
            regime_embedding=(0.01, 0.02, 2.0, 100.0),
            score=action,
        )
        for example, action in zip(examples, actions, strict=True)
    )

    report = validate_policy_acceptance(
        examples,
        predictions,
        PolicyAcceptanceConfig(
            max_dominant_bucket_ratio=0.50,
            max_active_regime_ratio=0.60,
        ),
    )

    assert report.passed
    assert report.acceptance_criteria["shadow_sharpe_positive"]
    assert report.acceptance_criteria["stable_action_distribution"]
    assert report.acceptance_criteria["monotonic_pnl_bucket_behavior"]
    assert report.acceptance_criteria["regime_action_stability"]
    assert report.acceptance_criteria["no_direct_pnl_optimization"]
    assert report.metrics["shadow_sharpe"] > 0.0
    assert report.metrics["regime_exposure"]["active_regime_count"] == 2
    assert report.metrics["regime_exposure"]["max_active_regime_ratio"] <= 0.60

    collapsed_predictions = tuple(
        PolicyPrediction(
            decision_ts=example.decision_ts,
            source=example.source,
            instrument_id=example.instrument_id,
            action=0.0,
            confidence=0.5,
            regime_embedding=(0.0, 0.0, 0.0, 0.0),
            score=0.0,
        )
        for example in examples
    )
    collapsed_report = validate_policy_acceptance(examples, collapsed_predictions)

    assert not collapsed_report.passed
    assert collapsed_report.acceptance_criteria["stable_action_distribution"] is False
    assert collapsed_report.acceptance_criteria["monotonic_pnl_bucket_behavior"] is False
    assert "unstable_action_distribution" in {
        failure.code for failure in collapsed_report.failures
    }


def test_policy_acceptance_fails_when_active_exposure_concentrates_in_one_regime() -> None:
    dataset = _causal_policy_dataset(row_count=8)
    examples = tuple(
        _copy_example(
            example,
            shadow_net_return=0.02,
            regime_key="crowded-regime" if idx < 6 else "quiet-regime",
        )
        for idx, example in enumerate(dataset.examples)
    )
    actions = (1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.0, 0.0)
    predictions = tuple(
        PolicyPrediction(
            decision_ts=example.decision_ts,
            source=example.source,
            instrument_id=example.instrument_id,
            action=action,
            confidence=0.8,
            regime_embedding=(0.0,),
            score=action,
        )
        for example, action in zip(examples, actions, strict=True)
    )

    report = validate_policy_acceptance(
        examples,
        predictions,
        PolicyAcceptanceConfig(max_active_regime_ratio=0.80),
    )

    assert not report.passed
    assert report.acceptance_criteria["regime_action_stability"] is False
    assert report.metrics["regime_exposure"]["active_regime_count"] == 1
    assert report.metrics["regime_exposure"]["max_active_regime_ratio"] == pytest.approx(1.0)
    assert "regime_exposure_concentration" in {
        failure.code for failure in report.failures
    }
