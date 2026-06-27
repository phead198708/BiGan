"""Polymarket policy training runner tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from bigan.v8.polymarket import (
    ACTION_VALUE_LABEL_ACTIONS,
    ACTION_VALUE_TARGET_FIELD,
    PRIMARY_POLICY_TARGET_ACTION_VALUE,
    PolymarketCorpusBuildConfig,
    PolymarketPolicyExample,
    PolymarketPolicyPrediction,
    PolymarketPolicyTrainingConfig,
    build_polymarket_btc_corpus,
    build_polymarket_ev_decisions,
    load_polymarket_policy_dataset,
    predict_polymarket_policy_examples,
    run_polymarket_policy_training,
    write_deterministic_polymarket_corpus_fixtures,
)
from bigan.v8.polymarket.contracts import looks_like_sha256
from bigan.v8.polymarket.training.action_family_eligibility import (
    build_action_family_counterfactual_prediction_sets,
)
from bigan.v8.polymarket.training.dataset import _split_examples


def test_training_dataset_loads_phase2_corpus_outputs(tmp_path: Path) -> None:
    corpus_dir = _build_corpus(tmp_path)
    config = PolymarketPolicyTrainingConfig(
        corpus_dir=corpus_dir,
        output_dir=tmp_path / "policy",
    )

    dataset = load_polymarket_policy_dataset(config)
    labels = _labels_by_decision_state(corpus_dir / "polymarket_label_rows.jsonl")

    assert len(dataset.examples) == 12
    assert dataset.feature_columns
    assert looks_like_sha256(dataset.feature_schema_hash)
    assert looks_like_sha256(dataset.label_schema_hash)
    assert looks_like_sha256(dataset.training_corpus_hash)
    assert looks_like_sha256(dataset.dataset_hash)
    assert dataset.split_metadata["split_strategy"] == "unique_decision_ts_temporal"
    assert dataset.split_metadata["strict_temporal_separation"] is True
    assert dataset.split_metadata["train_max_ts"] < dataset.split_metadata["validation_min_ts"]
    assert (
        dataset.split_metadata["validation_max_ts"]
        < dataset.split_metadata["shadow_min_ts"]
    )
    assert {example.market_family for example in dataset.examples} == {
        "btc_updown_5m",
        "btc_updown_15m",
        "btc_updown_1h",
    }
    dataset_payload = dataset.to_dict()
    for example in dataset.examples:
        assert example.feature_cutoff_ts <= example.decision_ts
        assert example.max_input_ts <= example.decision_ts
        assert 0.0 <= example.target_up_probability <= 1.0
        assert set(example.action_return_targets) == set(ACTION_VALUE_LABEL_ACTIONS)
        for action in ACTION_VALUE_LABEL_ACTIONS:
            assert example.action_return_targets[action] == labels[
                (example.market_id, example.decision_ts)
            ][action][ACTION_VALUE_TARGET_FIELD]
        assert example.best_policy_action in ACTION_VALUE_LABEL_ACTIONS
        assert example.best_action_expected_return >= example.second_best_action_expected_return
    assert dataset_payload["examples"][0]["action_return_targets"]


def test_feature_schema_hash_is_deterministic(tmp_path: Path) -> None:
    corpus_dir = _build_corpus(tmp_path)
    first = load_polymarket_policy_dataset(
        PolymarketPolicyTrainingConfig(
            corpus_dir=corpus_dir,
            output_dir=tmp_path / "first",
        )
    )
    second = load_polymarket_policy_dataset(
        PolymarketPolicyTrainingConfig(
            corpus_dir=corpus_dir,
            output_dir=tmp_path / "second",
        )
    )

    assert first.feature_schema_hash == second.feature_schema_hash
    assert first.dataset_hash == second.dataset_hash


def test_policy_split_keeps_shared_decision_ts_in_one_partition(tmp_path: Path) -> None:
    config = PolymarketPolicyTrainingConfig(
        corpus_dir=tmp_path / "corpus",
        output_dir=tmp_path / "policy",
        train_fraction=0.40,
        validation_fraction=0.30,
    )
    examples = tuple(
        _example(market_index=market_index, decision_ts=decision_ts)
        for decision_ts in (1_000, 2_000, 3_000, 4_000, 5_000)
        for market_index in (0, 1)
    )

    train, validation, shadow, metadata = _split_examples(examples, config)

    split_by_ts = {}
    for split_name, rows in (
        ("train", train),
        ("validation", validation),
        ("shadow", shadow),
    ):
        for row in rows:
            split_by_ts.setdefault(row.decision_ts, split_name)
            assert split_by_ts[row.decision_ts] == split_name
    assert metadata["train_max_ts"] < metadata["validation_min_ts"]
    assert metadata["validation_max_ts"] < metadata["shadow_min_ts"]
    assert len(train) == 4
    assert len(validation) == 2
    assert len(shadow) == 4


def test_training_runner_writes_required_artifacts_and_manifest(
    tmp_path: Path,
) -> None:
    corpus_dir = _build_corpus(tmp_path)
    result = run_polymarket_policy_training(
        PolymarketPolicyTrainingConfig(
            corpus_dir=corpus_dir,
            output_dir=tmp_path / "policy",
        )
    )

    expected = {
        "training_config",
        "dataset_profile",
        "model",
        "model_manifest",
        "calibration_report",
        "validation_report",
        "ev_threshold_report",
        "replay_report",
        "action_value_calibration",
        "action_value_signal_sanity_report",
        "action_value_signal_sanity_summary",
        "action_family_eligibility_report",
        "action_family_eligibility_summary",
        "hold_to_settlement_longshot_guard_report",
        "hold_to_settlement_longshot_guard_summary",
        "action_family_replay_variants_report",
        "action_family_replay_variants_summary",
        "action_family_counterfactual_replay_report",
        "action_family_counterfactual_replay_summary",
        "all_predictions",
        "predictions",
        "train_predictions",
        "validation_predictions",
        "shadow_predictions",
        "ev_decisions",
        "summary",
    }
    assert set(result.artifact_paths) == expected
    for name, path in result.artifact_paths.items():
        assert path.exists(), name
        assert looks_like_sha256(result.artifact_hashes[name])

    manifest = _read_json(result.artifact_paths["model_manifest"])
    profile = _read_json(result.artifact_paths["dataset_profile"])
    assert manifest["schema_version"] == "bigan-v8-polymarket-policy-v1"
    assert manifest["target"] == PRIMARY_POLICY_TARGET_ACTION_VALUE
    assert manifest["primary_policy_target"] == PRIMARY_POLICY_TARGET_ACTION_VALUE
    assert manifest["legacy_primary_policy_target"] == PRIMARY_POLICY_TARGET_ACTION_VALUE
    assert manifest["primary_policy_target_unit"] == "fixed_notional_net_pnl_per_notional"
    assert manifest["auxiliary_outcome_target"] == "resolved_up_probability"
    assert manifest["model_output"] == "action_expected_returns_with_p_up_auxiliary"
    assert "best_policy_action" in manifest["model_outputs"]
    assert manifest["outcome_probability_head_enabled"] is True
    assert manifest["action_value_head_enabled"] is True
    assert manifest["model_version"] == "polymarket_action_value_policy_v1"
    assert manifest["action_value_model_family"] == "feature_conditioned_action_return_model"
    assert manifest["fallback_action_value_model_family"] == "market_family_mean_baseline"
    assert manifest["feature_conditioned_action_value_model_enabled"] is True
    assert manifest["action_value_target_field"] == ACTION_VALUE_TARGET_FIELD
    assert manifest["fixed_notional_target_used"] is True
    assert manifest["action_value_calibration_artifact_path"] == (
        "polymarket_action_value_calibration.json"
    )
    assert looks_like_sha256(manifest["action_value_calibration_sha256"])
    assert manifest["action_value_calibration_artifact_used"] is True
    assert manifest["execution_uses_calibrated_action_value"] is True
    assert manifest["calibration_support_passed"] is True
    assert manifest["calibration_quality_passed"] is False
    assert manifest["calibration_quality_gates"][
        "shadow_calibrated_mae_not_worse"
    ] is False
    assert isinstance(
        manifest["calibration_quality_gates"][
            "high_score_bucket_min_support_passed"
        ],
        bool,
    )
    assert isinstance(
        manifest["calibration_quality_gates"][
            "high_score_bucket_realized_return_exceeds_buffer"
        ],
        bool,
    )
    assert manifest["shadow_mae_comparison"]["raw_mae"] == manifest[
        "calibration_quality_gates"
    ]["shadow_raw_mae"]
    assert (
        manifest["shadow_mae_comparison"]["action_level_calibrated_mae"]
        == manifest["calibration_quality_gates"][
            "shadow_action_level_calibrated_mae"
        ]
    )
    assert (
        manifest["shadow_mae_comparison"]["bucketed_calibrated_mae"]
        == manifest["calibration_quality_gates"]["shadow_bucketed_calibrated_mae"]
    )
    assert manifest["bucket_shrinkage_enabled"] is True
    assert manifest["bucket_shrinkage_prior"] > 0.0
    assert manifest["high_score_min_support"] >= 10
    assert manifest["high_score_execution_buffer"] == 0.015
    assert manifest["action_value_calibration_support_count"] > 0
    assert manifest["action_value_calibration_bucket_count"] >= len(ACTION_VALUE_LABEL_ACTIONS)
    assert isinstance(manifest["best_action_concentration_passed"], bool)
    assert isinstance(manifest["p_up_action_disagreement_within_limit"], bool)
    assert manifest["action_value_paper_decision_eligible"] is False
    assert "action_value_calibration_quality_failed" in manifest[
        "action_value_paper_decision_ineligible_reasons"
    ]
    assert manifest["action_value_signal_sanity_report"][
        "action_value_paper_decision_eligible"
    ] is False
    action_family_report = _read_json(
        result.artifact_paths["action_family_eligibility_report"]
    )
    assert action_family_report["schema_version"] == (
        "bigan-v8-polymarket-action-family-eligibility-v1"
    )
    assert action_family_report["out_of_sample_replay"] is True
    assert action_family_report["min_family_high_score_support"] >= 10
    assert action_family_report["family_high_score_execution_buffer"] == 0.015
    assert "action_family_paper_decision_eligible" in action_family_report
    assert "action_family_gate_results" in action_family_report
    assert "high_score_by_action" in action_family_report
    assert "high_score_by_action_family_side_price_time_raw_bucket" in action_family_report
    assert manifest["action_family_eligibility_report_path"] == (
        "action_family_eligibility_report.json"
    )
    assert manifest["action_family_paper_decision_eligible"] == action_family_report[
        "action_family_paper_decision_eligible"
    ]
    assert manifest["action_family_paper_decision_ineligible_reasons"] == (
        action_family_report["action_family_paper_decision_ineligible_reasons"]
    )
    if not action_family_report["action_family_paper_decision_eligible"]:
        for reason in action_family_report[
            "action_family_paper_decision_ineligible_reasons"
        ]:
            assert reason in manifest["action_value_paper_decision_ineligible_reasons"]
    longshot_report = _read_json(
        result.artifact_paths["hold_to_settlement_longshot_guard_report"]
    )
    assert longshot_report["schema_version"] == (
        "bigan-v8-polymarket-hold-to-settlement-longshot-guard-v1"
    )
    assert longshot_report["guard_enabled"] is True
    assert longshot_report["guard_mode"] == "block_to_no_trade"
    assert longshot_report["guard_reason_codes"] == [
        "hold_to_settlement_longshot_guard",
        "action_family_ineligible",
    ]
    assert manifest["hold_to_settlement_longshot_guard_enabled"] is True
    assert manifest["hold_to_settlement_longshot_guard_reason_codes"] == (
        longshot_report["guard_reason_codes"]
    )
    replay_variants = _read_json(
        result.artifact_paths["action_family_replay_variants_report"]
    )
    assert replay_variants["schema_version"] == (
        "bigan-v8-polymarket-action-family-replay-variants-v1"
    )
    assert [
        variant["variant"]
        for variant in replay_variants["variants"]
    ] == [
        "A_baseline_current_calibrated_policy_blocked",
        "B_hold_to_settlement_disabled",
        "C_sell_before_close_only",
        "D_hold_to_settlement_allowed_only_for_passed_buckets",
    ]
    assert [
        variant["threshold"]
        for variant in replay_variants["threshold_sweep_with_action_family_gates"]
    ] == [0.0, 0.03, 0.05]
    assert replay_variants["report_mode"] == "filtered_high_score_estimate"
    assert replay_variants["promotion_evidence_eligible"] is False
    counterfactual_replay = _read_json(
        result.artifact_paths["action_family_counterfactual_replay_report"]
    )
    assert counterfactual_replay["schema_version"] == (
        "bigan-v8-polymarket-action-family-counterfactual-replay-v1"
    )
    assert counterfactual_replay["report_mode"] == (
        "re_ranked_counterfactual_policy_replay"
    )
    assert counterfactual_replay["promotion_evidence_eligible"] is False
    assert [variant["variant"] for variant in counterfactual_replay["variants"]] == [
        "A_baseline_current_policy_with_runtime_guards",
        "B_hold_to_settlement_disabled_reranked",
        "C_sell_before_close_only_reranked",
        "D_hold_to_settlement_allowed_only_for_passed_buckets_reranked",
        "E_threshold_0.00_action_family_gates_reranked",
        "E_threshold_0.03_action_family_gates_reranked",
        "E_threshold_0.05_action_family_gates_reranked",
    ]
    for variant in counterfactual_replay["variants"]:
        assert variant["counterfactual_replay_mode"] == (
            "re_ranked_counterfactual_policy_replay"
        )
        assert variant["prediction_count"] == len(result.dataset.shadow_examples)
        assert variant["decision_count"] == len(result.dataset.shadow_examples)
        assert set(variant["artifact_paths"]) == {
            "decisions",
            "ev_threshold_report",
            "ledger_pnl_report",
            "policy_replay_report",
            "predictions",
        }
        for artifact_path in variant["artifact_paths"].values():
            assert (result.run_dir / artifact_path).exists()
        for artifact_hash in variant["artifact_hashes"].values():
            assert looks_like_sha256(artifact_hash)
    assert manifest["action_family_counterfactual_replay_report_path"] == (
        "action_family_counterfactual_replay_report.json"
    )
    assert looks_like_sha256(manifest["action_family_counterfactual_replay_sha256"])
    assert looks_like_sha256(manifest["action_family_eligibility_sha256"])
    assert looks_like_sha256(manifest["hold_to_settlement_longshot_guard_sha256"])
    assert looks_like_sha256(manifest["action_family_replay_variants_sha256"])
    action_value_calibration = _read_json(result.artifact_paths["action_value_calibration"])
    assert action_value_calibration["calibration_support_passed"] is True
    assert action_value_calibration["calibration_quality_passed"] is False
    assert action_value_calibration["calibration_fit_split"] == "validation"
    assert action_value_calibration["calibration_evaluation_split"] == "shadow"
    assert action_value_calibration["bucketed_calibration_enabled"] is True
    assert action_value_calibration["bucket_shrinkage_enabled"] is True
    assert action_value_calibration["bucket_shrinkage_prior"] > 0.0
    assert action_value_calibration["calibration_buckets"]
    low_support_bucket = next(
        bucket
        for bucket in action_value_calibration["calibration_buckets"].values()
        if bucket["support_count"] <= 2
    )
    assert abs(low_support_bucket["correction"]) <= abs(
        low_support_bucket["unshrunk_correction"]
    ) + 1e-12
    assert 0.0 < low_support_bucket["shrinkage_weight"] < 1.0
    assert manifest["action_value_feature_columns"]
    assert manifest["required_action_value_feature_columns"] == manifest[
        "action_value_feature_columns"
    ]
    assert profile["primary_policy_target"] == PRIMARY_POLICY_TARGET_ACTION_VALUE
    assert profile["action_value_target_field"] == ACTION_VALUE_TARGET_FIELD
    assert profile["fixed_notional_target_used"] is True
    assert profile["action_value_head_enabled"] is True
    assert profile["action_label_coverage_by_action"] == {
        action: len(result.dataset.examples) for action in ACTION_VALUE_LABEL_ACTIONS
    }
    assert set(manifest["market_families"]) == {
        "btc_updown_5m",
        "btc_updown_15m",
        "btc_updown_1h",
    }
    assert looks_like_sha256(manifest["model_sha256"])
    assert looks_like_sha256(manifest["training_corpus_hash"])
    assert looks_like_sha256(manifest["feature_schema_hash"])
    assert looks_like_sha256(manifest["label_schema_hash"])
    assert manifest["train_row_count"] > 0
    assert manifest["validation_row_count"] > 0
    assert manifest["shadow_row_count"] > 0
    assert manifest["train_max_ts"] < manifest["validation_min_ts"]
    assert manifest["validation_max_ts"] < manifest["shadow_min_ts"]
    assert manifest["strict_temporal_separation"] is True
    assert manifest["calibration_split"] == "validation"
    assert manifest["replay_split"] == "shadow"
    assert manifest["out_of_sample_replay"] is True
    assert manifest["direct_pnl_optimization"] is False
    assert manifest["trained_model_used"] is True
    assert manifest["policy_signal_source"] == "trained_model"
    assert manifest["synthetic_fixture_signal_used"] is False
    assert manifest["paper_replay_used_phase1_settlement_engine"] is True
    _assert_safe(manifest)


def test_feature_conditioned_action_returns_vary_by_state_within_family(
    tmp_path: Path,
) -> None:
    corpus_dir = _build_corpus(tmp_path)
    result = run_polymarket_policy_training(
        PolymarketPolicyTrainingConfig(
            corpus_dir=corpus_dir,
            output_dir=tmp_path / "policy",
        )
    )

    by_family = {}
    for prediction in result.predictions:
        by_family.setdefault(prediction.market_family, []).append(prediction)
    comparable = next(rows for rows in by_family.values() if len(rows) >= 2)
    first, second = comparable[0], comparable[-1]

    assert first.market_family == second.market_family
    assert first.features != second.features
    assert first.action_value_model_family == "feature_conditioned_action_return_model"
    assert first.feature_conditioned_action_value_model_enabled is True
    assert any(
        first.expected_return_by_action[action] != second.expected_return_by_action[action]
        for action in ACTION_VALUE_LABEL_ACTIONS
    )


def test_hold_to_settlement_disabled_counterfactual_reranks_to_sell_before_close(
    tmp_path: Path,
) -> None:
    action_returns = dict.fromkeys(ACTION_VALUE_LABEL_ACTIONS, -0.20)
    action_returns["NO_TRADE"] = 0.0
    action_returns["BUY_UP_HOLD_TO_SETTLEMENT"] = 0.20
    action_returns["BUY_UP_SELL_BEFORE_CLOSE"] = 0.12
    example = PolymarketPolicyExample(
        market_id="market-rerank",
        condition_id="condition-rerank",
        slug="btc-updown-rerank",
        market_family="btc_updown_5m",
        horizon_ms=300_000,
        decision_ts=1_000_000,
        feature_cutoff_ts=1_000_000,
        max_input_ts=1_000_000,
        features={
            "up_bid": 0.43,
            "up_ask": 0.45,
            "up_mid": 0.44,
            "down_bid": 0.53,
            "down_ask": 0.55,
            "down_mid": 0.54,
            "time_to_close_seconds": 120.0,
        },
        target_up_probability=1.0,
        resolved_outcome="UP",
        resolution_status="RESOLVED",
        action_return_targets=action_returns,
        best_policy_action="BUY_UP_HOLD_TO_SETTLEMENT",
        best_action_expected_return=0.20,
        second_best_action_expected_return=0.12,
        best_action_margin=0.08,
    )
    prediction = PolymarketPolicyPrediction(
        market_id=example.market_id,
        condition_id=example.condition_id,
        slug=example.slug,
        market_family=example.market_family,
        horizon_ms=example.horizon_ms,
        decision_ts=example.decision_ts,
        estimated_up_probability=0.70,
        confidence=0.90,
        score=0.20,
        calibration_bucket="test-bucket",
        model_version="test-action-value-model",
        feature_schema_hash="a" * 64,
        training_corpus_hash="b" * 64,
        features=dict(example.features),
        target_up_probability=example.target_up_probability,
        p_up_auxiliary=0.70,
        expected_return_by_action=action_returns,
        expected_return_no_trade=0.0,
        expected_return_buy_up_hold_to_settlement=0.20,
        expected_return_buy_down_hold_to_settlement=-0.20,
        expected_return_buy_up_sell_before_close=0.12,
        expected_return_buy_down_sell_before_close=-0.20,
        best_policy_action="BUY_UP_HOLD_TO_SETTLEMENT",
        best_action_expected_return=0.20,
        second_best_action_expected_return=0.12,
        best_action_margin=0.08,
        calibrated_expected_pnl_per_notional_by_action=action_returns,
        calibrated_best_policy_action="BUY_UP_HOLD_TO_SETTLEMENT",
        calibrated_expected_pnl_per_notional=0.20,
        calibrated_second_best_expected_pnl_per_notional=0.12,
        calibrated_action_margin=0.08,
        action_value_calibration_applied=True,
        action_value_calibration_id="c" * 64,
        calibration_support_count=10,
        calibration_bucket_count=len(ACTION_VALUE_LABEL_ACTIONS),
        policy_confidence=0.90,
        action_value_head_enabled=True,
        action_value_model_family="feature_conditioned_action_return_model",
        feature_conditioned_action_value_model_enabled=True,
    )

    variants = build_action_family_counterfactual_prediction_sets(
        examples=(example,),
        predictions=(prediction,),
        execution_buffer=0.015,
        thresholds=(),
    )
    hold_disabled = next(
        variant
        for variant in variants
        if variant["variant"] == "B_hold_to_settlement_disabled_reranked"
    )
    reranked_prediction = hold_disabled["predictions"][0]
    decisions = build_polymarket_ev_decisions(
        predictions=(reranked_prediction,),
        config=PolymarketPolicyTrainingConfig(
            corpus_dir=tmp_path / "corpus",
            output_dir=tmp_path / "policy",
            ev_threshold=0.015,
        ),
    )

    assert reranked_prediction.best_policy_action == "BUY_UP_SELL_BEFORE_CLOSE"
    assert (
        reranked_prediction.calibrated_best_policy_action
        == "BUY_UP_SELL_BEFORE_CLOSE"
    )
    assert reranked_prediction.calibrated_expected_pnl_per_notional == 0.12
    assert decisions[0].action == "BUY_UP"
    assert decisions[0].entry_policy_action == "BUY_UP_SELL_BEFORE_CLOSE"
    assert decisions[0].intended_exit_policy == "sell_before_close"
    assert decisions[0].planned_exit_before_ts is not None


def test_action_value_prediction_api_rejects_missing_features_by_default(
    tmp_path: Path,
) -> None:
    corpus_dir = _build_corpus(tmp_path)
    result = run_polymarket_policy_training(
        PolymarketPolicyTrainingConfig(
            corpus_dir=corpus_dir,
            output_dir=tmp_path / "policy",
        )
    )
    missing_feature = result.model.action_value_feature_columns[0]
    example = result.dataset.examples[0]
    sparse_features = dict(example.features)
    sparse_features.pop(missing_feature)
    sparse_example = replace(example, features=sparse_features)

    with pytest.raises(ValueError, match="action_value_feature_missing"):
        predict_polymarket_policy_examples(result.model, (sparse_example,))

    diagnostic_predictions = predict_polymarket_policy_examples(
        result.model,
        (sparse_example,),
        missing_feature_mode="train_mean_impute",
    )
    assert len(diagnostic_predictions) == 1
    assert diagnostic_predictions[0].best_policy_action in ACTION_VALUE_LABEL_ACTIONS


def _build_corpus(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw"
    corpus_dir = tmp_path / "corpus"
    write_deterministic_polymarket_corpus_fixtures(raw_dir)
    build_polymarket_btc_corpus(
        PolymarketCorpusBuildConfig(
            input_dir=raw_dir,
            output_dir=corpus_dir,
        )
    )
    return corpus_dir


def _example(*, market_index: int, decision_ts: int) -> PolymarketPolicyExample:
    return PolymarketPolicyExample(
        market_id=f"market-{decision_ts}-{market_index}",
        condition_id=f"condition-{decision_ts}-{market_index}",
        slug=f"btc-updown-{decision_ts}-{market_index}",
        market_family="btc_updown_15m",
        horizon_ms=900_000,
        decision_ts=decision_ts,
        feature_cutoff_ts=decision_ts,
        max_input_ts=decision_ts,
        features={
            "up_bid": 0.48,
            "up_ask": 0.52,
            "up_mid": 0.50,
            "down_bid": 0.48,
            "down_ask": 0.52,
            "down_mid": 0.50,
            "time_to_close_seconds": 120.0,
        },
        target_up_probability=1.0 if market_index == 0 else 0.0,
        resolved_outcome="UP" if market_index == 0 else "DOWN",
        resolution_status="RESOLVED",
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _labels_by_decision_state(path: Path) -> dict[tuple[str, int], dict[str, dict]]:
    labels: dict[tuple[str, int], dict[str, dict]] = {}
    for row in _read_jsonl(path):
        if row["action"] not in ACTION_VALUE_LABEL_ACTIONS:
            continue
        labels.setdefault((row["market_id"], row["decision_ts"]), {})[row["action"]] = row
    return labels


def _assert_safe(payload: dict) -> None:
    assert payload["paper_only"] is True
    assert payload["capital_at_risk"] is False
    assert payload["polymarket_write_enabled"] is False
    assert payload["wallet_signing_enabled"] is False
