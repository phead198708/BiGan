"""Polymarket policy calibration and validation report tests."""

from __future__ import annotations

import json
from pathlib import Path

from bigan.v8.polymarket import (
    ACTION_VALUE_LABEL_ACTIONS,
    PolymarketCorpusBuildConfig,
    PolymarketPolicyTrainingConfig,
    build_polymarket_btc_corpus,
    run_polymarket_policy_training,
    write_deterministic_polymarket_corpus_fixtures,
)


def test_model_predictions_are_probabilities_with_confidence(tmp_path: Path) -> None:
    result = _run_training(tmp_path)

    assert result.predictions
    for prediction in result.predictions:
        assert 0.0 <= prediction.estimated_up_probability <= 1.0
        assert 0.0 <= prediction.confidence <= 1.0
        assert prediction.model_version == result.model.model_version
        assert prediction.feature_schema_hash == result.dataset.feature_schema_hash
        assert prediction.training_corpus_hash == result.dataset.training_corpus_hash
        assert prediction.p_up_auxiliary == prediction.estimated_up_probability
        assert prediction.action_value_head_enabled is True
        assert prediction.outcome_probability_head_enabled is True
        assert prediction.action_value_model_family == "feature_conditioned_action_return_model"
        assert prediction.feature_conditioned_action_value_model_enabled is True
        assert set(prediction.expected_return_by_action) == set(ACTION_VALUE_LABEL_ACTIONS)
        assert prediction.best_policy_action in ACTION_VALUE_LABEL_ACTIONS
        assert prediction.best_action_expected_return is not None
        assert prediction.second_best_action_expected_return is not None
        assert prediction.best_action_margin is not None
        assert prediction.best_action_margin >= 0.0
        assert prediction.policy_confidence is not None
        assert 0.0 <= prediction.policy_confidence <= 1.0


def test_calibration_and_validation_reports_cover_families_and_time_buckets(
    tmp_path: Path,
) -> None:
    result = _run_training(tmp_path)
    calibration = _read_json(result.artifact_paths["calibration_report"])
    validation = _read_json(result.artifact_paths["validation_report"])

    assert calibration["primary_calibration_split"] == "validation"
    assert calibration["sample_count"] == len(result.dataset.validation_examples)
    assert calibration["calibration_error"] >= 0.0
    assert calibration["buckets"]
    assert calibration["train_calibration"]["sample_count"] == len(
        result.dataset.train_examples
    )
    assert calibration["validation_calibration"]["sample_count"] == len(
        result.dataset.validation_examples
    )
    assert calibration["shadow_calibration"]["sample_count"] == len(
        result.dataset.shadow_examples
    )
    assert (
        calibration["primary_calibration"]["sample_count"]
        == calibration["validation_calibration"]["sample_count"]
    )
    assert calibration["sample_counts_by_split"] == {
        "train": len(result.dataset.train_examples),
        "validation": len(result.dataset.validation_examples),
        "shadow": len(result.dataset.shadow_examples),
    }
    assert validation["evaluation_split"] == "validation"
    assert validation["out_of_sample_validation"] is True
    assert validation["validation"]["sample_count"] > 0
    assert validation["validation"]["logloss"] >= 0.0
    assert validation["validation"]["brier_score"] >= 0.0
    assert set(validation["metrics_by_market_family"]) == {
        "btc_updown_5m",
        "btc_updown_15m",
        "btc_updown_1h",
    }
    assert (
        sum(
            row["sample_count"]
            for row in validation["metrics_by_market_family"].values()
        )
        == len(result.dataset.validation_examples)
    )
    assert set(validation["metrics_by_time_to_close_bucket"]) == {
        "0-30s",
        "30-60s",
        "1-3m",
        "3-5m",
        "5-15m",
        "15m+",
    }
    assert (
        sum(
            row["sample_count"]
            for row in validation["metrics_by_time_to_close_bucket"].values()
        )
        == len(result.dataset.validation_examples)
    )
    assert validation["model_is_calibrated_better_than_naive_baseline"] is True
    _assert_safe(calibration)
    _assert_safe(validation)


def test_all_json_artifacts_preserve_paper_only_safety_flags(tmp_path: Path) -> None:
    result = _run_training(tmp_path)

    for name, path in result.artifact_paths.items():
        if path.suffix == ".md":
            continue
        if path.suffix == ".jsonl":
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            assert rows, name
            for row in rows:
                _assert_safe(row)
            continue
        _assert_safe(_read_json(path))


def _run_training(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    corpus_dir = tmp_path / "corpus"
    write_deterministic_polymarket_corpus_fixtures(raw_dir)
    build_polymarket_btc_corpus(
        PolymarketCorpusBuildConfig(
            input_dir=raw_dir,
            output_dir=corpus_dir,
        )
    )
    return run_polymarket_policy_training(
        PolymarketPolicyTrainingConfig(
            corpus_dir=corpus_dir,
            output_dir=tmp_path / "policy",
        )
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_safe(payload: dict) -> None:
    assert payload["paper_only"] is True
    assert payload["capital_at_risk"] is False
    assert payload["polymarket_write_enabled"] is False
    assert payload["wallet_signing_enabled"] is False
