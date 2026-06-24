"""Polymarket policy training runner tests."""

from __future__ import annotations

import json
from pathlib import Path

from bigan.v8.polymarket import (
    ACTION_VALUE_LABEL_ACTIONS,
    PRIMARY_POLICY_TARGET_ACTION_VALUE,
    PolymarketCorpusBuildConfig,
    PolymarketPolicyExample,
    PolymarketPolicyTrainingConfig,
    build_polymarket_btc_corpus,
    load_polymarket_policy_dataset,
    run_polymarket_policy_training,
    write_deterministic_polymarket_corpus_fixtures,
)
from bigan.v8.polymarket.contracts import looks_like_sha256
from bigan.v8.polymarket.training.dataset import _split_examples


def test_training_dataset_loads_phase2_corpus_outputs(tmp_path: Path) -> None:
    corpus_dir = _build_corpus(tmp_path)
    config = PolymarketPolicyTrainingConfig(
        corpus_dir=corpus_dir,
        output_dir=tmp_path / "policy",
    )

    dataset = load_polymarket_policy_dataset(config)

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
    assert manifest["auxiliary_outcome_target"] == "resolved_up_probability"
    assert manifest["model_output"] == "action_expected_returns_with_p_up_auxiliary"
    assert "best_policy_action" in manifest["model_outputs"]
    assert manifest["outcome_probability_head_enabled"] is True
    assert manifest["action_value_head_enabled"] is True
    assert profile["primary_policy_target"] == PRIMARY_POLICY_TARGET_ACTION_VALUE
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


def _assert_safe(payload: dict) -> None:
    assert payload["paper_only"] is True
    assert payload["capital_at_risk"] is False
    assert payload["polymarket_write_enabled"] is False
    assert payload["wallet_signing_enabled"] is False
