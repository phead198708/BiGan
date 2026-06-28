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
from bigan.v8.polymarket.contracts import canonical_json_sha256, looks_like_sha256
from bigan.v8.polymarket.training.action_family_eligibility import (
    build_action_family_counterfactual_prediction_sets,
)
from bigan.v8.polymarket.training.dataset import _split_examples
from bigan.v8.polymarket.training.model_ranking_diagnostics import (
    build_model_ranking_candidate_comparison,
)
from bigan.v8.polymarket.training.sell_before_close_diagnostics import (
    build_sell_before_close_p_up_disagreement_diagnostic_report,
)


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
        for action in ("BUY_UP_SELL_BEFORE_CLOSE", "BUY_DOWN_SELL_BEFORE_CLOSE"):
            assert action in example.sell_before_close_execution_class_targets
            assert action in example.sell_before_close_theoretical_return_targets
            assert action in example.sell_before_close_executable_return_targets
            assert action in example.sell_before_close_execution_gap_targets
            assert action in example.sell_before_close_queue_fill_probability_targets
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


def test_policy_dataset_rejects_fixed_terminal_only_sell_before_close_labels(
    tmp_path: Path,
) -> None:
    corpus_dir = _build_corpus(tmp_path)
    labels_path = corpus_dir / "polymarket_label_rows.jsonl"
    labels = _read_jsonl(labels_path)
    for row in labels:
        if row["action"].endswith("SELL_BEFORE_CLOSE"):
            row.pop("sell_before_close_label_schema_version", None)
            row["sell_before_close_exit_path"] = {
                "label_source": "fixed_terminal_bid_only"
            }
            break
    _write_jsonl(labels_path, labels)

    with pytest.raises(ValueError, match="executable exit schema"):
        load_polymarket_policy_dataset(
            PolymarketPolicyTrainingConfig(
                corpus_dir=corpus_dir,
                output_dir=tmp_path / "policy",
            )
        )


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
        "model_ranking_error_report",
        "model_ranking_error_summary",
        "model_ranking_candidate_comparison",
        "model_ranking_candidate_comparison_summary",
        "action_representation_diagnostic_report",
        "action_representation_diagnostic_summary",
        "ranking_overlay_zero_entry_diagnostic_report",
        "ranking_overlay_zero_entry_diagnostic_summary",
        "source_model_eligibility_report",
        "source_model_eligibility_summary",
        "sell_before_close_p_up_disagreement_diagnostic_report",
        "sell_before_close_p_up_disagreement_diagnostic_summary",
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
    assert manifest["sell_before_close_label_schema_version"] == (
        "bigan-v8-polymarket-sell-before-close-executable-exit-v1"
    )
    assert manifest["sell_before_close_fixed_terminal_bid_only_labels_allowed"] is False
    assert manifest["sell_before_close_label_gate_passed"] is True
    assert manifest["sell_before_close_execution_class_counts"]
    assert profile["sell_before_close_label_schema_version"] == (
        "bigan-v8-polymarket-sell-before-close-executable-exit-v1"
    )
    assert profile["sell_before_close_fixed_terminal_bid_only_labels_allowed"] is False
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
    ranking_error = _read_json(result.artifact_paths["model_ranking_error_report"])
    assert ranking_error["schema_version"] == (
        "bigan-v8-polymarket-model-ranking-error-v1"
    )
    assert set(ranking_error["diagnostic_splits"]) == {"validation", "shadow"}
    for split_name, expected_count in (
        ("validation", len(result.dataset.validation_examples)),
        ("shadow", len(result.dataset.shadow_examples)),
    ):
        split = ranking_error[split_name]
        assert split["sample_count"] == expected_count
        assert 0.0 <= split["top_1_action_hit_rate"] <= 1.0
        assert 0.0 <= split["top_2_action_hit_rate"] <= 1.0
        assert 0.0 <= split["top_3_action_hit_rate"] <= 1.0
        assert split["rows"]
        first_row = split["rows"][0]
        for field_name in (
            "calibrated_best_policy_action",
            "realized_best_action",
            "rank_of_realized_best_action_under_calibrated_scores",
            "score_spread_selected_minus_realized_best",
            "selected_action_realized_return",
            "oracle_best_action_realized_return",
            "regret",
        ):
            assert field_name in first_row
        assert set(split["breakdowns"]) == {
            "action_family",
            "side",
            "price_bucket",
            "time_to_close_bucket",
            "raw_score_bucket",
            "market_family",
        }
    candidate_comparison = _read_json(
        result.artifact_paths["model_ranking_candidate_comparison"]
    )
    candidate_comparison_id = candidate_comparison[
        "model_ranking_candidate_comparison_id"
    ]
    candidate_comparison_payload = dict(candidate_comparison)
    candidate_comparison_payload.pop("model_ranking_candidate_comparison_id")
    assert canonical_json_sha256(candidate_comparison_payload) == (
        candidate_comparison_id
    )
    assert candidate_comparison["schema_version"] == (
        "bigan-v8-polymarket-model-ranking-candidate-comparison-v1"
    )
    assert candidate_comparison["candidate_count"] >= 6
    assert candidate_comparison["source_model_candidate_eligible"] is False
    assert candidate_comparison["requires_promotion_replay_gate"] is True
    assert candidate_comparison["paper_run_resume_allowed"] is False
    assert candidate_comparison["paper_run_resume_blocked_reason"] == (
        "promotion_replay_gate_required"
    )
    assert {
        "A_current_model_baseline",
        "B_family_specific_calibration_only",
        "C_action_specific_calibration_with_family_gates",
        "D_pairwise_rank_correction",
        "E_action_family_prior_penalty",
        "F_live_eligible_feature_subset_retrain_proxy",
        "I_sell_before_close_only_source_candidate",
    }.issubset(set(candidate_comparison["candidate_names"]))
    for candidate in candidate_comparison["candidates"]:
        for field_name in (
            "shadow_raw_mae",
            "shadow_calibrated_mae",
            "high_score_support_count",
            "high_score_realized_return_mean",
            "high_score_realized_return_sum",
            "action_family_gates",
            "source_model_eligible",
            "ineligible_reason_codes",
            "enabled_action_families",
            "disabled_action_families",
            "enabled_actions",
            "disabled_actions",
            "candidate_scoped_p_up_action_disagreement_rate",
            "candidate_scoped_p_up_action_disagreement_within_limit",
            "candidate_scoped_action_family_gate_results",
            "candidate_scoped_high_score_support_count",
            "candidate_scoped_high_score_realized_return_mean",
            "candidate_scoped_high_score_realized_return_sum",
        ):
            assert field_name in candidate
    sell_only_candidate = _candidate_by_name(
        candidate_comparison,
        "I_sell_before_close_only_source_candidate",
    )
    assert sell_only_candidate["enabled_action_families"] == ["SELL_BEFORE_CLOSE"]
    assert sell_only_candidate["disabled_action_families"] == ["HOLD_TO_SETTLEMENT"]
    assert sell_only_candidate["enabled_actions"] == [
        "NO_TRADE",
        "BUY_UP_SELL_BEFORE_CLOSE",
        "BUY_DOWN_SELL_BEFORE_CLOSE",
    ]
    assert sell_only_candidate["disabled_actions"] == [
        "BUY_UP_HOLD_TO_SETTLEMENT",
        "BUY_DOWN_HOLD_TO_SETTLEMENT",
    ]
    source_eligibility = _read_json(
        result.artifact_paths["source_model_eligibility_report"]
    )
    source_eligibility_id = source_eligibility["source_model_eligibility_report_id"]
    source_eligibility_payload = dict(source_eligibility)
    source_eligibility_payload.pop("source_model_eligibility_report_id")
    assert canonical_json_sha256(source_eligibility_payload) == source_eligibility_id
    assert source_eligibility["schema_version"] == (
        "bigan-v8-polymarket-source-model-eligibility-v1"
    )
    assert source_eligibility["source_model_eligible"] is False
    assert source_eligibility["source_model_candidate_eligible"] is False
    assert source_eligibility["requires_promotion_replay_gate"] is True
    assert source_eligibility["paper_run_resume_allowed"] is False
    assert source_eligibility["paper_run_resume_blocked_reason"] == (
        "promotion_replay_gate_required"
    )
    assert source_eligibility["hard_gates"]["calibration_quality_passed"] is False
    assert source_eligibility["candidate_count"] == candidate_comparison["candidate_count"]
    assert len(source_eligibility["candidate_scoped_eligibility_summary"]) == (
        candidate_comparison["candidate_count"]
    )
    assert manifest["candidate_scoped_source_model_eligibility_summary"] == (
        source_eligibility["candidate_scoped_eligibility_summary"]
    )
    assert manifest[
        "sell_before_close_p_up_disagreement_diagnostic_report_path"
    ] == "sell_before_close_p_up_disagreement_diagnostic_report.json"
    assert looks_like_sha256(
        manifest["sell_before_close_p_up_disagreement_diagnostic_sha256"]
    )
    diagnostic = _read_json(
        result.artifact_paths[
            "sell_before_close_p_up_disagreement_diagnostic_report"
        ]
    )
    assert diagnostic["schema_version"] == (
        "bigan-v8-polymarket-sell-before-close-p-up-disagreement-diagnostic-v1"
    )
    assert diagnostic["candidate_name"] == (
        "I_sell_before_close_only_source_candidate"
    )
    assert diagnostic["diagnostic_only"] is True
    assert diagnostic["promotion_evidence_eligible"] is False
    assert diagnostic["paper_run_resume_allowed"] is False
    assert manifest["sell_before_close_p_up_disagreement_interpretation"] in {
        "likely_model_direction_error",
        "likely_auxiliary_p_up_action_value_semantic_mismatch",
        "mixed_evidence",
        "insufficient_evidence",
    }
    assert manifest["sell_before_close_p_up_disagreement_diagnostic_summary"] == (
        source_eligibility["sell_before_close_p_up_disagreement_diagnostic_summary"]
    )
    assert candidate_comparison[
        "sell_before_close_p_up_disagreement_diagnostic_summary"
    ] == source_eligibility[
        "sell_before_close_p_up_disagreement_diagnostic_summary"
    ]
    counterfactual_replay = _read_json(
        result.artifact_paths["action_family_counterfactual_replay_report"]
    )
    assert counterfactual_replay[
        "sell_before_close_p_up_disagreement_diagnostic_summary"
    ] == source_eligibility[
        "sell_before_close_p_up_disagreement_diagnostic_summary"
    ]
    assert candidate_comparison["candidate_artifact_count"] >= 2
    exported_names = {
        artifact["candidate_name"]
        for artifact in candidate_comparison["candidate_artifacts"]
    }
    assert {
        "G_bucketed_lcb_rank_selector",
        "H_positive_bucket_rank_selector",
    }.issubset(exported_names)
    for artifact in candidate_comparison["candidate_artifacts"]:
        assert set(artifact["artifact_paths"]) == {
            "manifest",
            "predictions",
            "ranking_overlay",
        }
        for artifact_path in artifact["artifact_paths"].values():
            assert (result.run_dir / artifact_path).exists()
        for artifact_hash in artifact["artifact_hashes"].values():
            assert looks_like_sha256(artifact_hash)
        if artifact["candidate_name"] in {
            "G_bucketed_lcb_rank_selector",
            "H_positive_bucket_rank_selector",
        }:
            overlay = _read_json(
                result.run_dir / artifact["artifact_paths"]["ranking_overlay"]
            )
            candidate_manifest = _read_json(
                result.run_dir / artifact["artifact_paths"]["manifest"]
            )
            assert overlay["fit_split"] == "validation"
            assert overlay["evaluation_split"] == "shadow"
            assert overlay["uses_shadow_for_fit"] is False
            assert overlay["shrinkage_prior_support"] == 10
            assert overlay["shrinkage_prior_mean"] == 0.0
            assert "bucket_evidence_weight" in overlay
            assert "model_score_weight" in overlay
            assert candidate_manifest["ranking_overlay_fit_split"] == "validation"
            assert candidate_manifest["ranking_overlay_evaluation_split"] == "shadow"
            assert candidate_manifest["ranking_overlay_uses_shadow_split"] is False
            assert candidate_manifest["ranking_overlay_min_bucket_support"] >= 3
            assert candidate_manifest["ranking_overlay_min_family_support"] == 10
            assert candidate_manifest["ranking_overlay_shrinkage_prior_support"] == 10
            assert candidate_manifest["ranking_overlay_shrinkage_prior_mean"] == 0.0
            assert "ranking_overlay_score_combination" in candidate_manifest
            assert "ranking_overlay_bucket_evidence_weight" in candidate_manifest
            assert "ranking_overlay_model_score_weight" in candidate_manifest
    assert source_eligibility["candidate_artifacts"] == candidate_comparison[
        "candidate_artifacts"
    ]
    assert manifest["model_ranking_error_report_path"] == (
        "model_ranking_error_report.json"
    )
    assert looks_like_sha256(manifest["model_ranking_error_report_sha256"])
    assert manifest["model_ranking_candidate_comparison_path"] == (
        "model_ranking_candidate_comparison.json"
    )
    assert looks_like_sha256(manifest["model_ranking_candidate_comparison_sha256"])
    action_representation = _read_json(
        result.artifact_paths["action_representation_diagnostic_report"]
    )
    action_representation_id = action_representation[
        "action_representation_diagnostic_report_id"
    ]
    action_representation_payload = dict(action_representation)
    action_representation_payload.pop("action_representation_diagnostic_report_id")
    assert canonical_json_sha256(action_representation_payload) == (
        action_representation_id
    )
    assert action_representation["schema_version"] == (
        "bigan-v8-polymarket-action-representation-diagnostic-v1"
    )
    assert action_representation["diagnostic_only"] is True
    assert action_representation["promotion_evidence_eligible"] is False
    assert action_representation["source_model_candidate_eligible"] is False
    assert action_representation["paper_run_resume_allowed"] is False
    assert action_representation["fine_action_family_definition"] == (
        "side|intended_exit_policy|price_bucket|time_to_close_bucket"
    )
    assert action_representation["label_exit_path_assessment"][
        "sell_before_close_exit_path_coarse"
    ] is False
    assert action_representation["label_exit_path_assessment"][
        "sell_before_close_exit_path_is_fixed_terminal_bid"
    ] is False
    assert action_representation["label_exit_path_assessment"][
        "uses_intraround_exit_opportunity_model"
    ] is True
    assert action_representation["label_exit_path_assessment"][
        "uses_queue_fill_probability_model"
    ] is True
    assert action_representation["label_exit_path_assessment"][
        "compares_theoretical_vs_executable_exit_return"
    ] is True
    assert "single_terminal_exit_bid_path" not in action_representation[
        "label_exit_path_assessment"
    ]["coarse_exit_path_risk_codes"]
    for split_name in ("validation", "shadow"):
        split = action_representation[split_name]
        assert split["sell_before_close_summary"]["support_count"] > 0
        assert "action_family_summary" in split
        assert "fine_action_family_summary" in split
        assert "side_exit_policy_price_time_summary" in split
        assert "sell_before_close_negative_contributors" in split
        assert "sell_before_close_positive_supported_buckets" in split
        assert "top_negative_high_score_sell_before_close_examples" in split
        for row in split["fine_action_family_summary"]:
            assert "fine_action_family" in row
            assert "unique_market_count" in row
            assert "realized_trade_return_mean" in row
            assert "theoretical_terminal_bid_return_mean" in row
            assert "realized_executable_sell_before_close_return_mean" in row
            assert "execution_gap_return_mean" in row
            assert "sell_before_close_execution_class_distribution" in row
        for row in split["top_negative_high_score_sell_before_close_examples"]:
            assert "fine_action_family" in row
            assert "calibrated_score" in row
            assert "realized_return" in row
            assert "time_to_close_seconds" in row
    assert result.action_representation_diagnostic_report == action_representation
    assert manifest["action_representation_diagnostic_report_path"] == (
        "action_representation_diagnostic_report.json"
    )
    assert looks_like_sha256(manifest["action_representation_diagnostic_sha256"])
    zero_entry_report = _read_json(
        result.artifact_paths["ranking_overlay_zero_entry_diagnostic_report"]
    )
    zero_entry_report_id = zero_entry_report[
        "ranking_overlay_zero_entry_diagnostic_report_id"
    ]
    zero_entry_payload = dict(zero_entry_report)
    zero_entry_payload.pop("ranking_overlay_zero_entry_diagnostic_report_id")
    assert canonical_json_sha256(zero_entry_payload) == zero_entry_report_id
    assert zero_entry_report["schema_version"] == (
        "bigan-v8-polymarket-ranking-overlay-zero-entry-diagnostic-v1"
    )
    assert zero_entry_report["diagnostic_only"] is True
    assert zero_entry_report["promotion_evidence_eligible"] is False
    assert zero_entry_report["source_model_candidate_eligible"] is False
    assert zero_entry_report["paper_run_resume_allowed"] is False
    assert zero_entry_report["uses_shadow_for_fit"] is False
    assert {
        "G_bucketed_lcb_rank_selector",
        "H_positive_bucket_rank_selector",
    } == set(zero_entry_report["candidate_names"])
    assert len(zero_entry_report["diagnostic_sweeps"]) == 54
    for row in zero_entry_report["diagnostic_sweeps"]:
        assert row["diagnostic_only"] is True
        assert row["source_model_candidate_eligible"] is False
        assert row["promotion_eligible"] is False
        assert row["paper_run_resume_allowed"] is False
        assert row["paper_only"] is True
        assert row["capital_at_risk"] is False
    for candidate in zero_entry_report["candidates"]:
        assert candidate["prediction_count"] == len(result.dataset.shadow_examples)
        assert candidate["action_count_considered"] == (
            len(result.dataset.shadow_examples)
            * (len(ACTION_VALUE_LABEL_ACTIONS) - 1)
        )
        assert candidate["non_no_trade_candidate_count"] == (
            candidate["action_count_considered"]
        )
        assert candidate["selected_non_no_trade_count"] >= 0
        assert "bucket_missing_count" in candidate
        assert "family_missing_count" in candidate
        assert "bucket_support_failed_count" in candidate
        assert "family_support_failed_count" in candidate
        assert "bucket_lcb_or_mean_failed_count" in candidate
        assert "family_lcb_or_mean_failed_count" in candidate
        assert "bucket_sum_failed_count" in candidate
        assert "passed_bucket_and_family_count" in candidate
        assert set(candidate["grouped_summaries"]) == {
            "action",
            "action_family",
            "fine_action_family",
            "intended_exit_policy",
            "side",
            "price_bucket",
            "time_to_close_bucket",
            "raw_score_bucket",
            "market_family",
        }
        assert "top_near_pass_buckets" in candidate
        assert "top_near_pass_families" in candidate
        assert candidate["source_model_candidate_eligible"] is False
        assert candidate["promotion_eligible"] is False
    assert result.ranking_overlay_zero_entry_diagnostic_report == zero_entry_report
    assert manifest["ranking_overlay_zero_entry_diagnostic_report_path"] == (
        "ranking_overlay_zero_entry_diagnostic_report.json"
    )
    assert looks_like_sha256(
        manifest["ranking_overlay_zero_entry_diagnostic_sha256"]
    )
    assert manifest["source_model_eligibility_report_path"] == (
        "source_model_eligibility_report.json"
    )
    assert looks_like_sha256(manifest["source_model_eligibility_report_sha256"])
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
    assert "fine_action_family_gate_results" in action_family_report
    assert "high_score_by_action" in action_family_report
    assert "high_score_by_fine_action_family" in action_family_report
    assert "high_score_by_action_family_side_price_time_raw_bucket" in action_family_report
    assert "high_score_by_side_exit_policy_price_time_bucket" in action_family_report
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
        "I_sell_before_close_only_source_candidate",
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


def test_bucketed_overlay_uses_validation_only_and_selects_supported_positive_bucket(
    tmp_path: Path,
) -> None:
    validation_examples, raw_validation, calibrated_validation = (
        _overlay_examples_and_predictions(
            count=10,
            positive_sell_return=0.20,
            negative_hold_return=-0.20,
        )
    )
    shadow_examples, raw_shadow, calibrated_shadow = _overlay_examples_and_predictions(
        count=1,
        positive_sell_return=-0.90,
        negative_hold_return=0.90,
        start_ts=20_000,
    )

    comparison = build_model_ranking_candidate_comparison(
        validation_examples=validation_examples,
        raw_validation_predictions=raw_validation,
        calibrated_validation_predictions=calibrated_validation,
        shadow_examples=shadow_examples,
        raw_shadow_predictions=raw_shadow,
        calibrated_shadow_predictions=calibrated_shadow,
        execution_buffer=0.015,
    )

    for candidate_name in (
        "G_bucketed_lcb_rank_selector",
        "H_positive_bucket_rank_selector",
    ):
        candidate = _candidate_by_name(comparison, candidate_name)
        assert candidate["ranking_overlay_fit_split"] == "validation"
        assert candidate["ranking_overlay_evaluation_split"] == "shadow"
        assert candidate["ranking_overlay_uses_shadow_split"] is False
        assert candidate["ranking_overlay"]["shrinkage_prior_support"] == 10
        assert candidate["ranking_overlay"]["shrinkage_prior_mean"] == 0.0
        assert candidate["ranking_overlay_shrinkage_prior_support"] == 10
        assert candidate["ranking_overlay_shrinkage_prior_mean"] == 0.0
        assert candidate["candidate_predictions"][0][
            "calibrated_best_policy_action"
        ] == "BUY_UP_SELL_BEFORE_CLOSE"


def test_bucketed_overlay_blocks_low_support_validation_buckets(tmp_path: Path) -> None:
    validation_examples, raw_validation, calibrated_validation = (
        _overlay_examples_and_predictions(
            count=2,
            positive_sell_return=0.20,
            negative_hold_return=-0.20,
        )
    )
    shadow_examples, raw_shadow, calibrated_shadow = _overlay_examples_and_predictions(
        count=1,
        positive_sell_return=0.20,
        negative_hold_return=-0.20,
        start_ts=20_000,
    )

    comparison = build_model_ranking_candidate_comparison(
        validation_examples=validation_examples,
        raw_validation_predictions=raw_validation,
        calibrated_validation_predictions=calibrated_validation,
        shadow_examples=shadow_examples,
        raw_shadow_predictions=raw_shadow,
        calibrated_shadow_predictions=calibrated_shadow,
        execution_buffer=0.015,
    )

    for candidate_name in (
        "G_bucketed_lcb_rank_selector",
        "H_positive_bucket_rank_selector",
    ):
        candidate = _candidate_by_name(comparison, candidate_name)
        assert candidate["candidate_predictions"][0][
            "calibrated_best_policy_action"
        ] == "NO_TRADE"


def test_bucketed_overlay_requires_buffer_positive_validation_buckets(
    tmp_path: Path,
) -> None:
    validation_examples, raw_validation, calibrated_validation = (
        _overlay_examples_and_predictions(
            count=10,
            positive_sell_return=0.01,
            negative_hold_return=-0.20,
        )
    )
    shadow_examples, raw_shadow, calibrated_shadow = _overlay_examples_and_predictions(
        count=1,
        positive_sell_return=0.20,
        negative_hold_return=-0.20,
        start_ts=20_000,
    )

    comparison = build_model_ranking_candidate_comparison(
        validation_examples=validation_examples,
        raw_validation_predictions=raw_validation,
        calibrated_validation_predictions=calibrated_validation,
        shadow_examples=shadow_examples,
        raw_shadow_predictions=raw_shadow,
        calibrated_shadow_predictions=calibrated_shadow,
        execution_buffer=0.015,
    )

    for candidate_name in (
        "G_bucketed_lcb_rank_selector",
        "H_positive_bucket_rank_selector",
    ):
        candidate = _candidate_by_name(comparison, candidate_name)
        assert candidate["candidate_predictions"][0][
            "calibrated_best_policy_action"
        ] == "NO_TRADE"


def test_bucketed_overlay_blocks_negative_validation_buckets(tmp_path: Path) -> None:
    validation_examples, raw_validation, calibrated_validation = (
        _overlay_examples_and_predictions(
            count=10,
            positive_sell_return=-0.20,
            negative_hold_return=-0.10,
        )
    )
    shadow_examples, raw_shadow, calibrated_shadow = _overlay_examples_and_predictions(
        count=1,
        positive_sell_return=0.20,
        negative_hold_return=-0.20,
        start_ts=20_000,
    )

    comparison = build_model_ranking_candidate_comparison(
        validation_examples=validation_examples,
        raw_validation_predictions=raw_validation,
        calibrated_validation_predictions=calibrated_validation,
        shadow_examples=shadow_examples,
        raw_shadow_predictions=raw_shadow,
        calibrated_shadow_predictions=calibrated_shadow,
        execution_buffer=0.015,
    )

    for candidate_name in (
        "G_bucketed_lcb_rank_selector",
        "H_positive_bucket_rank_selector",
    ):
        candidate = _candidate_by_name(comparison, candidate_name)
        assert candidate["candidate_predictions"][0][
            "calibrated_best_policy_action"
        ] == "NO_TRADE"


def test_sell_before_close_source_candidate_ignores_disabled_hold_blockers(
    tmp_path: Path,
) -> None:
    comparison = _sell_before_close_candidate_comparison(
        positive_sell_return=0.20,
        hold_return=-0.50,
        selected_sell_action="BUY_UP_SELL_BEFORE_CLOSE",
        p_up_auxiliary=0.70,
    )
    candidate = _candidate_by_name(
        comparison,
        "I_sell_before_close_only_source_candidate",
    )

    assert candidate["enabled_action_families"] == ["SELL_BEFORE_CLOSE"]
    assert candidate["disabled_action_families"] == ["HOLD_TO_SETTLEMENT"]
    assert candidate["source_model_candidate_eligible"] is True
    assert candidate["action_family_paper_decision_eligible"] is True
    assert candidate["candidate_scoped_action_family_gate_results"][
        "SELL_BEFORE_CLOSE"
    ]["gate_passed"] is True
    assert "HOLD_TO_SETTLEMENT" not in candidate[
        "candidate_scoped_action_family_gate_results"
    ]
    assert "HOLD_TO_SETTLEMENT" in candidate[
        "candidate_scoped_disabled_action_family_gate_results"
    ]
    assert "hold_to_settlement_high_score_unprofitable" not in candidate[
        "ineligible_reason_codes"
    ]
    assert "buy_up_hold_to_settlement_unprofitable" not in candidate[
        "ineligible_reason_codes"
    ]
    assert "buy_down_hold_to_settlement_unprofitable" not in candidate[
        "ineligible_reason_codes"
    ]
    assert comparison["source_model_candidate_eligible"] is True


def test_sell_before_close_candidate_scopes_p_up_disagreement_to_enabled_actions(
    tmp_path: Path,
) -> None:
    comparison = _sell_before_close_candidate_comparison(
        positive_sell_return=0.20,
        hold_return=-0.50,
        selected_sell_action="BUY_UP_SELL_BEFORE_CLOSE",
        p_up_auxiliary=0.70,
    )
    candidate = _candidate_by_name(
        comparison,
        "I_sell_before_close_only_source_candidate",
    )
    baseline = _candidate_by_name(comparison, "A_current_model_baseline")

    assert baseline["p_up_action_disagreement_rate"] == pytest.approx(1.0)
    assert candidate["candidate_scoped_p_up_action_disagreement_denominator"] == 12
    assert candidate["candidate_scoped_p_up_action_disagreement_rate"] == pytest.approx(
        2 / 12
    )
    assert candidate["candidate_scoped_p_up_action_disagreement_within_limit"] is True
    assert candidate["source_model_candidate_eligible"] is True
    assert "p_up_action_disagreement_excessive" not in candidate[
        "ineligible_reason_codes"
    ]


def test_sell_before_close_candidate_fails_closed_when_sell_family_fails(
    tmp_path: Path,
) -> None:
    comparison = _sell_before_close_candidate_comparison(
        positive_sell_return=-0.20,
        hold_return=-0.50,
        selected_sell_action="BUY_UP_SELL_BEFORE_CLOSE",
        p_up_auxiliary=0.70,
    )
    candidate = _candidate_by_name(
        comparison,
        "I_sell_before_close_only_source_candidate",
    )

    assert candidate["source_model_candidate_eligible"] is False
    assert candidate["action_family_paper_decision_eligible"] is False
    assert candidate["candidate_scoped_high_score_realized_return_mean"] == pytest.approx(
        -0.20
    )
    assert "action_family_high_score_unprofitable" in candidate[
        "ineligible_reason_codes"
    ]
    assert "action_value_calibration_quality_failed" in candidate[
        "ineligible_reason_codes"
    ]


def test_sell_before_close_candidate_does_not_resume_without_promotion_replay(
    tmp_path: Path,
) -> None:
    comparison = _sell_before_close_candidate_comparison(
        positive_sell_return=0.20,
        hold_return=-0.50,
        selected_sell_action="BUY_UP_SELL_BEFORE_CLOSE",
        p_up_auxiliary=0.70,
    )
    candidate = _candidate_by_name(
        comparison,
        "I_sell_before_close_only_source_candidate",
    )

    assert candidate["source_model_candidate_eligible"] is True
    assert candidate["requires_promotion_replay_gate"] is True
    assert candidate["paper_run_resume_allowed"] is False
    assert comparison["paper_run_resume_allowed"] is False
    assert comparison["paper_run_resume_blocked_reason"] == (
        "promotion_replay_gate_required"
    )


def test_sell_before_close_p_up_disagreement_diagnostic_is_candidate_scoped() -> None:
    shadow_examples, comparison, counterfactual_replays = (
        _sell_before_close_diagnostic_inputs()
    )

    report = build_sell_before_close_p_up_disagreement_diagnostic_report(
        shadow_examples=shadow_examples,
        model_ranking_candidate_comparison=comparison,
        action_family_counterfactual_replays=counterfactual_replays,
        pnl_notional=0.20,
    )

    assert report["diagnostic_only"] is True
    assert report["promotion_evidence_eligible"] is False
    assert report["paper_run_resume_allowed"] is False
    assert report["candidate_name"] == "I_sell_before_close_only_source_candidate"
    assert report["row_count"] == 12
    assert {
        row["selected_action"] for row in report["row_level_diagnostics"]
    } == {
        "BUY_UP_SELL_BEFORE_CLOSE",
        "BUY_DOWN_SELL_BEFORE_CLOSE",
    }
    assert report["summary"]["disagreed_support_count"] == 10
    assert report["summary"]["agreed_support_count"] == 2
    assert report["summary"][
        "sell_before_close_disagreed_trade_pnl_sum"
    ] == pytest.approx(0.30)
    assert report["summary"][
        "sell_before_close_disagreed_settlement_pnl_sum"
    ] == pytest.approx(-0.50)
    assert report["summary"]["sell_before_close_disagreed_total_pnl_sum"] == (
        pytest.approx(-0.20)
    )
    assert report["p_up_disagreement_interpretation"] == (
        "likely_auxiliary_p_up_action_value_semantic_mismatch"
    )
    first_row = report["row_level_diagnostics"][0]
    for field_name in (
        "market_id",
        "slug",
        "decision_ts",
        "market_end_ts",
        "seconds_to_close",
        "selected_action",
        "selected_side",
        "p_up",
        "p_down",
        "p_up_direction",
        "action_side",
        "p_up_action_disagrees",
        "calibrated_action_score",
        "second_best_action",
        "best_action_margin",
        "entry_ask",
        "entry_bid",
        "exit_bid",
        "exit_ts",
        "sell_before_close_execution_class",
        "queue_fill_probability_estimate",
        "executable_liquidity_notional",
        "realized_trade_return",
        "settlement_return",
        "realized_total_return",
        "trade_pnl_contribution",
        "settlement_pnl_contribution",
        "total_pnl_contribution",
        "counterfactual_replay_variant",
        "reason_codes",
    ):
        assert field_name in first_row
    assert report["comparison_tables"][
        "high_p_up_disagreement_rows_with_positive_trade_pnl_negative_settlement_pnl"
    ]
    assert report["counterfactual_replay_attribution"][
        "positions_opened_but_not_closed_before_settlement"
    ] == 3


def _sell_before_close_diagnostic_inputs() -> tuple[
    tuple[PolymarketPolicyExample, ...],
    dict,
    tuple[dict, ...],
]:
    validation_examples, raw_validation, calibrated_validation = (
        _sell_before_close_candidate_examples_and_predictions(
            count=12,
            start_ts=50_000,
            selected_sell_action="BUY_DOWN_SELL_BEFORE_CLOSE",
            selected_sell_realized_return=0.20,
            hold_realized_return=-0.50,
            p_up_auxiliary=0.70,
        )
    )
    shadow_examples, raw_shadow, calibrated_shadow = (
        _sell_before_close_candidate_examples_and_predictions(
            count=12,
            start_ts=60_000,
            selected_sell_action="BUY_DOWN_SELL_BEFORE_CLOSE",
            selected_sell_realized_return=0.20,
            hold_realized_return=-0.50,
            p_up_auxiliary=0.70,
        )
    )
    validation_examples = tuple(
        _with_sell_before_close_diagnostic_targets(example)
        for example in validation_examples
    )
    shadow_examples = tuple(
        _with_sell_before_close_diagnostic_targets(example)
        for example in shadow_examples
    )
    comparison = build_model_ranking_candidate_comparison(
        validation_examples=validation_examples,
        raw_validation_predictions=raw_validation,
        calibrated_validation_predictions=calibrated_validation,
        shadow_examples=shadow_examples,
        raw_shadow_predictions=raw_shadow,
        calibrated_shadow_predictions=calibrated_shadow,
        execution_buffer=0.015,
    )
    return (
        shadow_examples,
        comparison,
        (
            {
                "variant": "I_sell_before_close_only_source_candidate",
                "summary": {
                    "entry_decision_count": 12,
                    "action_counts": {
                        "BUY_UP": 2,
                        "BUY_DOWN": 10,
                        "SELL_UP": 5,
                        "SELL_DOWN": 4,
                    },
                    "reason_counts": {
                        "hold_threshold_not_met": 3,
                        "low_confidence": 1,
                    },
                    "realized_trade_pnl": 1.0,
                    "settlement_pnl": -2.0,
                    "total_polymarket_pnl": -1.0,
                    "settlement_event_count": 3,
                },
            },
        ),
    )


def _with_sell_before_close_diagnostic_targets(
    example: PolymarketPolicyExample,
) -> PolymarketPolicyExample:
    action_returns = {
        "NO_TRADE": 0.0,
        "BUY_UP_HOLD_TO_SETTLEMENT": -0.50,
        "BUY_DOWN_HOLD_TO_SETTLEMENT": -0.50,
        "BUY_UP_SELL_BEFORE_CLOSE": 0.08,
        "BUY_DOWN_SELL_BEFORE_CLOSE": -0.10,
    }
    trade_returns = {
        "NO_TRADE": 0.0,
        "BUY_UP_HOLD_TO_SETTLEMENT": 0.0,
        "BUY_DOWN_HOLD_TO_SETTLEMENT": 0.0,
        "BUY_UP_SELL_BEFORE_CLOSE": 0.05,
        "BUY_DOWN_SELL_BEFORE_CLOSE": 0.15,
    }
    settlement_returns = {
        "NO_TRADE": 0.0,
        "BUY_UP_HOLD_TO_SETTLEMENT": -0.50,
        "BUY_DOWN_HOLD_TO_SETTLEMENT": -0.50,
        "BUY_UP_SELL_BEFORE_CLOSE": 0.03,
        "BUY_DOWN_SELL_BEFORE_CLOSE": -0.25,
    }
    ranked = _rank_action_returns(action_returns)
    features = {
        **example.features,
        "up_executable_bid_notional": 0.20,
        "down_executable_bid_notional": 0.20,
    }
    return replace(
        example,
        features=features,
        action_return_targets=action_returns,
        realized_trade_return_targets=trade_returns,
        settlement_return_targets=settlement_returns,
        action_is_positive_targets={
            action: value > 0.0 for action, value in action_returns.items()
        },
        sell_before_close_execution_class_targets={
            "BUY_UP_SELL_BEFORE_CLOSE": "realizable_sell_before_close",
            "BUY_DOWN_SELL_BEFORE_CLOSE": "realizable_sell_before_close",
        },
        sell_before_close_queue_fill_probability_targets={
            "BUY_UP_SELL_BEFORE_CLOSE": 0.90,
            "BUY_DOWN_SELL_BEFORE_CLOSE": 0.80,
        },
        sell_before_close_exit_bid_targets={
            "BUY_UP_SELL_BEFORE_CLOSE": 0.55,
            "BUY_DOWN_SELL_BEFORE_CLOSE": 0.60,
        },
        sell_before_close_executable_liquidity_notional_targets={
            "BUY_UP_SELL_BEFORE_CLOSE": 0.20,
            "BUY_DOWN_SELL_BEFORE_CLOSE": 0.20,
        },
        sell_before_close_exit_path_targets={
            "BUY_UP_SELL_BEFORE_CLOSE": {
                "best_executable_exit_ts": example.decision_ts + 30_000,
            },
            "BUY_DOWN_SELL_BEFORE_CLOSE": {
                "best_executable_exit_ts": example.decision_ts + 30_000,
            },
        },
        sell_before_close_label_uses_executable_exit_path_targets={
            "BUY_UP_SELL_BEFORE_CLOSE": True,
            "BUY_DOWN_SELL_BEFORE_CLOSE": True,
        },
        best_policy_action=ranked[0][0],
        best_action_expected_return=ranked[0][1],
        second_best_action_expected_return=ranked[1][1],
        best_action_margin=ranked[0][1] - ranked[1][1],
    )


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


def _overlay_examples_and_predictions(
    *,
    count: int,
    positive_sell_return: float,
    negative_hold_return: float,
    start_ts: int = 10_000,
) -> tuple[
    tuple[PolymarketPolicyExample, ...],
    tuple[PolymarketPolicyPrediction, ...],
    tuple[PolymarketPolicyPrediction, ...],
]:
    examples = []
    predictions = []
    for index in range(count):
        action_returns = dict.fromkeys(ACTION_VALUE_LABEL_ACTIONS, -0.30)
        action_returns["NO_TRADE"] = 0.0
        action_returns["BUY_UP_SELL_BEFORE_CLOSE"] = positive_sell_return
        action_returns["BUY_DOWN_SELL_BEFORE_CLOSE"] = positive_sell_return
        action_returns["BUY_DOWN_HOLD_TO_SETTLEMENT"] = negative_hold_return
        decision_ts = start_ts + index
        features = {
            "up_bid": 0.43,
            "up_ask": 0.45,
            "up_mid": 0.44,
            "down_bid": 0.53,
            "down_ask": 0.55,
            "down_mid": 0.54,
            "time_to_close_seconds": 120.0,
        }
        example = PolymarketPolicyExample(
            market_id=f"overlay-market-{decision_ts}",
            condition_id=f"overlay-condition-{decision_ts}",
            slug=f"overlay-slug-{decision_ts}",
            market_family="btc_updown_5m",
            horizon_ms=300_000,
            decision_ts=decision_ts,
            feature_cutoff_ts=decision_ts,
            max_input_ts=decision_ts,
            features=features,
            target_up_probability=1.0,
            resolved_outcome="UP",
            resolution_status="RESOLVED",
            action_return_targets=action_returns,
            best_policy_action=_rank_action_returns(action_returns)[0][0],
            best_action_expected_return=_rank_action_returns(action_returns)[0][1],
            second_best_action_expected_return=_rank_action_returns(action_returns)[1][1],
            best_action_margin=(
                _rank_action_returns(action_returns)[0][1]
                - _rank_action_returns(action_returns)[1][1]
            ),
        )
        prediction = _overlay_prediction(
            example=example,
            action_returns={
                "NO_TRADE": 0.0,
                "BUY_UP_HOLD_TO_SETTLEMENT": -0.10,
                "BUY_DOWN_HOLD_TO_SETTLEMENT": 0.50,
                "BUY_UP_SELL_BEFORE_CLOSE": 0.10,
                "BUY_DOWN_SELL_BEFORE_CLOSE": -0.10,
            },
        )
        examples.append(example)
        predictions.append(prediction)
    return tuple(examples), tuple(predictions), tuple(predictions)


def _overlay_prediction(
    *,
    example: PolymarketPolicyExample,
    action_returns: dict[str, float],
) -> PolymarketPolicyPrediction:
    return PolymarketPolicyPrediction(
        market_id=example.market_id,
        condition_id=example.condition_id,
        slug=example.slug,
        market_family=example.market_family,
        horizon_ms=example.horizon_ms,
        decision_ts=example.decision_ts,
        estimated_up_probability=0.70,
        confidence=0.90,
        score=0.50,
        calibration_bucket="overlay-test",
        model_version="overlay-test-model",
        feature_schema_hash="a" * 64,
        training_corpus_hash="b" * 64,
        features=dict(example.features),
        target_up_probability=example.target_up_probability,
        p_up_auxiliary=0.70,
        expected_return_by_action=action_returns,
        expected_return_no_trade=action_returns["NO_TRADE"],
        expected_return_buy_up_hold_to_settlement=action_returns[
            "BUY_UP_HOLD_TO_SETTLEMENT"
        ],
        expected_return_buy_down_hold_to_settlement=action_returns[
            "BUY_DOWN_HOLD_TO_SETTLEMENT"
        ],
        expected_return_buy_up_sell_before_close=action_returns[
            "BUY_UP_SELL_BEFORE_CLOSE"
        ],
        expected_return_buy_down_sell_before_close=action_returns[
            "BUY_DOWN_SELL_BEFORE_CLOSE"
        ],
        best_policy_action="BUY_DOWN_HOLD_TO_SETTLEMENT",
        best_action_expected_return=0.50,
        second_best_action_expected_return=0.10,
        best_action_margin=0.40,
        calibrated_expected_pnl_per_notional_by_action=action_returns,
        calibrated_best_policy_action="BUY_DOWN_HOLD_TO_SETTLEMENT",
        calibrated_expected_pnl_per_notional=0.50,
        calibrated_second_best_expected_pnl_per_notional=0.10,
        calibrated_action_margin=0.40,
        action_value_calibration_applied=True,
        action_value_calibration_id="c" * 64,
        calibration_support_count=10,
        calibration_bucket_count=len(ACTION_VALUE_LABEL_ACTIONS),
        policy_confidence=0.90,
        action_value_head_enabled=True,
        action_value_model_family="feature_conditioned_action_return_model",
        feature_conditioned_action_value_model_enabled=True,
    )


def _sell_before_close_candidate_comparison(
    *,
    positive_sell_return: float,
    hold_return: float,
    selected_sell_action: str,
    p_up_auxiliary: float,
) -> dict:
    validation_examples, raw_validation, calibrated_validation = (
        _sell_before_close_candidate_examples_and_predictions(
            count=12,
            start_ts=30_000,
            selected_sell_action=selected_sell_action,
            selected_sell_realized_return=positive_sell_return,
            hold_realized_return=hold_return,
            p_up_auxiliary=p_up_auxiliary,
        )
    )
    shadow_examples, raw_shadow, calibrated_shadow = (
        _sell_before_close_candidate_examples_and_predictions(
            count=12,
            start_ts=40_000,
            selected_sell_action=selected_sell_action,
            selected_sell_realized_return=positive_sell_return,
            hold_realized_return=hold_return,
            p_up_auxiliary=p_up_auxiliary,
        )
    )
    return build_model_ranking_candidate_comparison(
        validation_examples=validation_examples,
        raw_validation_predictions=raw_validation,
        calibrated_validation_predictions=calibrated_validation,
        shadow_examples=shadow_examples,
        raw_shadow_predictions=raw_shadow,
        calibrated_shadow_predictions=calibrated_shadow,
        execution_buffer=0.015,
    )


def _sell_before_close_candidate_examples_and_predictions(
    *,
    count: int,
    start_ts: int,
    selected_sell_action: str,
    selected_sell_realized_return: float,
    hold_realized_return: float,
    p_up_auxiliary: float,
) -> tuple[
    tuple[PolymarketPolicyExample, ...],
    tuple[PolymarketPolicyPrediction, ...],
    tuple[PolymarketPolicyPrediction, ...],
]:
    other_sell_action = (
        "BUY_DOWN_SELL_BEFORE_CLOSE"
        if selected_sell_action == "BUY_UP_SELL_BEFORE_CLOSE"
        else "BUY_UP_SELL_BEFORE_CLOSE"
    )
    examples = []
    predictions = []
    for index in range(count):
        decision_ts = start_ts + index
        example = _example(market_index=index % 2, decision_ts=decision_ts)
        active_sell_action = (
            other_sell_action if index % 6 == 0 else selected_sell_action
        )
        inactive_sell_action = (
            selected_sell_action if active_sell_action == other_sell_action else other_sell_action
        )
        action_targets = {
            "NO_TRADE": 0.0,
            "BUY_UP_HOLD_TO_SETTLEMENT": hold_realized_return,
            "BUY_DOWN_HOLD_TO_SETTLEMENT": hold_realized_return,
            selected_sell_action: selected_sell_realized_return,
            other_sell_action: selected_sell_realized_return,
        }
        ranked_targets = _rank_action_returns(action_targets)
        example = replace(
            example,
            action_return_targets=action_targets,
            best_policy_action=ranked_targets[0][0],
            best_action_expected_return=ranked_targets[0][1],
            second_best_action_expected_return=ranked_targets[1][1],
            best_action_margin=ranked_targets[0][1] - ranked_targets[1][1],
        )
        action_scores = {
            "NO_TRADE": 0.0,
            "BUY_UP_HOLD_TO_SETTLEMENT": 0.30,
            "BUY_DOWN_HOLD_TO_SETTLEMENT": 0.50,
            active_sell_action: 0.20,
            inactive_sell_action: 0.10,
        }
        prediction = _scoped_candidate_prediction(
            example=example,
            action_returns=action_scores,
            p_up_auxiliary=p_up_auxiliary,
        )
        examples.append(example)
        predictions.append(prediction)
    return tuple(examples), tuple(predictions), tuple(predictions)


def _scoped_candidate_prediction(
    *,
    example: PolymarketPolicyExample,
    action_returns: dict[str, float],
    p_up_auxiliary: float,
) -> PolymarketPolicyPrediction:
    ranked = _rank_action_returns(action_returns)
    best_action, best_return = ranked[0]
    second_return = ranked[1][1]
    return PolymarketPolicyPrediction(
        market_id=example.market_id,
        condition_id=example.condition_id,
        slug=example.slug,
        market_family=example.market_family,
        horizon_ms=example.horizon_ms,
        decision_ts=example.decision_ts,
        estimated_up_probability=p_up_auxiliary,
        confidence=0.90,
        score=best_return,
        calibration_bucket="scoped-source-candidate-test",
        model_version="scoped-source-candidate-test-model",
        feature_schema_hash="a" * 64,
        training_corpus_hash="b" * 64,
        features=dict(example.features),
        target_up_probability=example.target_up_probability,
        p_up_auxiliary=p_up_auxiliary,
        expected_return_by_action=action_returns,
        expected_return_no_trade=action_returns["NO_TRADE"],
        expected_return_buy_up_hold_to_settlement=action_returns[
            "BUY_UP_HOLD_TO_SETTLEMENT"
        ],
        expected_return_buy_down_hold_to_settlement=action_returns[
            "BUY_DOWN_HOLD_TO_SETTLEMENT"
        ],
        expected_return_buy_up_sell_before_close=action_returns[
            "BUY_UP_SELL_BEFORE_CLOSE"
        ],
        expected_return_buy_down_sell_before_close=action_returns[
            "BUY_DOWN_SELL_BEFORE_CLOSE"
        ],
        best_policy_action=best_action,
        best_action_expected_return=best_return,
        second_best_action_expected_return=second_return,
        best_action_margin=best_return - second_return,
        calibrated_expected_pnl_per_notional_by_action=action_returns,
        calibrated_best_policy_action=best_action,
        calibrated_expected_pnl_per_notional=best_return,
        calibrated_second_best_expected_pnl_per_notional=second_return,
        calibrated_action_margin=best_return - second_return,
        action_value_calibration_applied=True,
        action_value_calibration_id="d" * 64,
        calibration_support_count=12,
        calibration_bucket_count=len(ACTION_VALUE_LABEL_ACTIONS),
        policy_confidence=0.90,
        action_value_head_enabled=True,
        action_value_model_family="feature_conditioned_action_return_model",
        feature_conditioned_action_value_model_enabled=True,
    )


def _candidate_by_name(comparison: dict, candidate_name: str) -> dict:
    return next(
        candidate
        for candidate in comparison["candidates"]
        if candidate["candidate_name"] == candidate_name
    )


def _rank_action_returns(action_returns: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(action_returns.items(), key=lambda item: (-float(item[1]), item[0]))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


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
