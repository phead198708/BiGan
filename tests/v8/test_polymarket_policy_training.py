"""Polymarket policy training runner tests."""

from __future__ import annotations

import json
from pathlib import Path

from bigan.v8.polymarket import (
    PolymarketCorpusBuildConfig,
    PolymarketPolicyTrainingConfig,
    build_polymarket_btc_corpus,
    load_polymarket_policy_dataset,
    run_polymarket_policy_training,
    write_deterministic_polymarket_corpus_fixtures,
)
from bigan.v8.polymarket.contracts import looks_like_sha256


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
    assert {example.market_family for example in dataset.examples} == {
        "btc_updown_5m",
        "btc_updown_15m",
        "btc_updown_1h",
    }
    for example in dataset.examples:
        assert example.feature_cutoff_ts <= example.decision_ts
        assert example.max_input_ts <= example.decision_ts
        assert 0.0 <= example.target_up_probability <= 1.0


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
        "predictions",
        "ev_decisions",
        "summary",
    }
    assert set(result.artifact_paths) == expected
    for name, path in result.artifact_paths.items():
        assert path.exists(), name
        assert looks_like_sha256(result.artifact_hashes[name])

    manifest = _read_json(result.artifact_paths["model_manifest"])
    assert manifest["schema_version"] == "bigan-v8-polymarket-policy-v1"
    assert manifest["target"] == "resolved_up"
    assert manifest["model_output"] == "estimated_up_probability"
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


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_safe(payload: dict) -> None:
    assert payload["paper_only"] is True
    assert payload["capital_at_risk"] is False
    assert payload["polymarket_write_enabled"] is False
    assert payload["wallet_signing_enabled"] is False
