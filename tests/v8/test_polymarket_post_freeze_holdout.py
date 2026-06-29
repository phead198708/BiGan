"""Post-freeze holdout validation tests for the frozen M selector."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from bigan.v8.polymarket import (
    PolymarketCorpusBuildConfig,
    PolymarketPolicyTrainingConfig,
    build_polymarket_btc_corpus,
    run_polymarket_policy_training,
    write_deterministic_polymarket_corpus_fixtures,
)
from bigan.v8.polymarket.contracts import canonical_json_sha256, looks_like_sha256
from bigan.v8.polymarket.training.post_freeze_holdout import (
    FROZEN_M_SELECTOR_BASELINE_COMMIT,
    FROZEN_M_SELECTOR_METHOD,
    M_POST_FREEZE_HOLDOUT_SCHEMA_VERSION,
    PolymarketPostFreezeHoldoutConfig,
    run_polymarket_m_post_freeze_holdout_validation,
)


def test_post_freeze_holdout_blocks_same_lineage_before_prediction(
    tmp_path: Path,
) -> None:
    corpus_dir = _build_corpus(tmp_path / "source")
    training = run_polymarket_policy_training(
        PolymarketPolicyTrainingConfig(
            corpus_dir=corpus_dir,
            output_dir=tmp_path / "training",
        )
    )
    _patch_frozen_manifest_lineage(training.run_dir, corpus_dir, training.dataset.dataset_hash)

    result = run_polymarket_m_post_freeze_holdout_validation(
        PolymarketPostFreezeHoldoutConfig(
            frozen_model_dir=training.run_dir,
            frozen_corpus_dir=corpus_dir,
            holdout_corpus_dir=corpus_dir,
            output_dir=tmp_path / "holdout",
        )
    )

    report = result.report
    assert report["schema_version"] == M_POST_FREEZE_HOLDOUT_SCHEMA_VERSION
    assert report["validation_status"] == "blocked_fail_closed"
    assert report["prediction_attempted"] is False
    assert report["true_post_freeze_holdout"] is False
    assert "holdout_not_strictly_after_frozen_training_window" in report[
        "reason_codes"
    ]
    assert "holdout_market_ids_overlap_frozen_training_corpus" in report[
        "reason_codes"
    ]
    assert "holdout_dataset_hash_matches_frozen_dataset" in report["reason_codes"]
    assert report["selected_entry_count"] == 0
    assert report["replay_entry_count"] == 0
    assert report["source_model_candidate_eligible"] is False
    assert report["promotion_evidence_eligible"] is False
    assert report["paper_run_resume_allowed"] is False
    assert report["#146_start_allowed"] is False
    assert report["#134_resume_allowed"] is False
    assert result.artifact_paths["report"].exists()
    assert result.artifact_paths["summary"].exists()


def test_post_freeze_holdout_runs_later_disjoint_corpus_with_frozen_weights(
    tmp_path: Path,
) -> None:
    corpus_dir = _build_corpus(tmp_path / "source")
    training = run_polymarket_policy_training(
        PolymarketPolicyTrainingConfig(
            corpus_dir=corpus_dir,
            output_dir=tmp_path / "training",
        )
    )
    _patch_frozen_manifest_lineage(training.run_dir, corpus_dir, training.dataset.dataset_hash)
    frozen_max_ts = max(example.decision_ts for example in training.dataset.examples)
    holdout_corpus_dir = _copy_later_disjoint_corpus(
        source=corpus_dir,
        destination=tmp_path / "later_holdout_corpus",
        min_after_ts=frozen_max_ts,
    )

    result = run_polymarket_m_post_freeze_holdout_validation(
        PolymarketPostFreezeHoldoutConfig(
            frozen_model_dir=training.run_dir,
            frozen_corpus_dir=corpus_dir,
            holdout_corpus_dir=holdout_corpus_dir,
            output_dir=tmp_path / "holdout",
        )
    )

    report = result.report
    report_id = report["m_post_freeze_holdout_validation_report_id"]
    payload = dict(report)
    payload.pop("m_post_freeze_holdout_validation_report_id")
    assert canonical_json_sha256(payload) == report_id
    assert report["validation_status"] == "completed"
    assert report["prediction_attempted"] is True
    assert report["true_post_freeze_holdout"] is True
    assert report["baseline_selector_commit"] == FROZEN_M_SELECTOR_BASELINE_COMMIT
    assert report["selector_method"] == FROZEN_M_SELECTOR_METHOD
    assert report["selector_weights_unchanged_from_baseline"] is True
    assert report["rank_weight_tuning_allowed"] is False
    assert report["rank_weight_tuning_performed"] is False
    assert report["holdout_feedback_used_for_tuning"] is False
    assert report["p_up_side_alignment_filter_enabled"] is False
    assert report["p_up_side_alignment_diagnostic_enabled"] is True
    assert report["provenance"]["holdout_strictly_after_frozen"] is True
    assert report["provenance"]["market_id_disjoint"] is True
    assert report["provenance"]["dataset_hash_changed"] is True
    assert report["provenance"]["frozen_model_sha256_matches_manifest"] is True
    assert report["provenance"]["frozen_dataset_hash_matches_manifest"] is True
    assert report["provenance"]["frozen_split_hash_matches_manifest"] is True
    assert report["provenance"][
        "frozen_corpus_dir_matches_frozen_training_lineage"
    ] is True
    assert report["selected_exit_decision_count"] == 0
    assert report["replay_entry_reconciliation"]["reconciled"] is True
    assert set(report["replay_pnl_by_side"]) == {"UP", "DOWN"}
    assert "rank_score_component_summary" in report
    assert report["source_model_candidate_eligible"] is False
    assert report["promotion_evidence_eligible"] is False
    assert report["paper_run_resume_allowed"] is False
    assert report["#146_start_allowed"] is False
    assert report["#134_resume_allowed"] is False
    manifest = _read_json(result.artifact_paths["manifest"])
    assert looks_like_sha256(manifest["artifact_hashes"]["report"])
    assert looks_like_sha256(manifest["artifact_hashes"]["summary"])


def test_post_freeze_holdout_blocks_wrong_frozen_corpus_before_prediction(
    tmp_path: Path,
) -> None:
    corpus_dir = _build_corpus(tmp_path / "source")
    training = run_polymarket_policy_training(
        PolymarketPolicyTrainingConfig(
            corpus_dir=corpus_dir,
            output_dir=tmp_path / "training",
        )
    )
    _patch_frozen_manifest_lineage(training.run_dir, corpus_dir, training.dataset.dataset_hash)
    frozen_max_ts = max(example.decision_ts for example in training.dataset.examples)
    wrong_frozen_corpus_dir = _copy_later_disjoint_corpus(
        source=corpus_dir,
        destination=tmp_path / "wrong_frozen_corpus",
        min_after_ts=frozen_max_ts,
    )
    holdout_corpus_dir = _copy_later_disjoint_corpus(
        source=corpus_dir,
        destination=tmp_path / "later_holdout_corpus",
        min_after_ts=frozen_max_ts + 10_000_000,
    )

    result = run_polymarket_m_post_freeze_holdout_validation(
        PolymarketPostFreezeHoldoutConfig(
            frozen_model_dir=training.run_dir,
            frozen_corpus_dir=wrong_frozen_corpus_dir,
            holdout_corpus_dir=holdout_corpus_dir,
            output_dir=tmp_path / "holdout",
        )
    )

    report = result.report
    assert report["validation_status"] == "blocked_fail_closed"
    assert report["prediction_attempted"] is False
    assert report["source_model_candidate_eligible"] is False
    assert report["promotion_evidence_eligible"] is False
    assert report["#146_start_allowed"] is False
    assert report["#134_resume_allowed"] is False
    assert report["provenance"]["frozen_model_sha256_matches_manifest"] is True
    assert report["provenance"]["frozen_dataset_hash_matches_manifest"] is False
    assert report["provenance"][
        "frozen_corpus_dir_matches_frozen_training_lineage"
    ] is False
    assert "frozen_dataset_hash_mismatch_manifest" in report["reason_codes"]
    assert "frozen_corpus_dir_not_frozen_training_lineage" in report["reason_codes"]
    assert report["selected_entry_count"] == 0
    assert report["replay_entry_count"] == 0


def _build_corpus(root: Path) -> Path:
    raw_dir = root / "raw"
    corpus_dir = root / "corpus"
    write_deterministic_polymarket_corpus_fixtures(raw_dir)
    build_polymarket_btc_corpus(
        PolymarketCorpusBuildConfig(
            input_dir=raw_dir,
            output_dir=corpus_dir,
        )
    )
    return corpus_dir


def _patch_frozen_manifest_lineage(
    run_dir: Path,
    corpus_dir: Path,
    dataset_hash: str,
) -> None:
    manifest_path = run_dir / "polymarket_policy_model_manifest.json"
    manifest = _read_json(manifest_path)
    split = _read_json(corpus_dir / "polymarket_train_shadow_split.json")
    manifest["policy_dataset_hash"] = dataset_hash
    manifest["split_hash"] = split["split_hash"]
    manifest["train_dataset_hash"] = split["train_dataset_hash"]
    manifest["shadow_dataset_hash"] = split["shadow_dataset_hash"]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _copy_later_disjoint_corpus(
    *,
    source: Path,
    destination: Path,
    min_after_ts: int,
) -> Path:
    shutil.copytree(source, destination)
    feature_rows = _read_jsonl(destination / "polymarket_feature_rows.jsonl")
    source_min_ts = min(int(row["decision_ts"]) for row in feature_rows)
    offset = int(min_after_ts) + 60_000 - source_min_ts
    suffix = "-post-freeze-holdout"
    for path in destination.glob("*.jsonl"):
        rows = [_transform_payload(row, offset=offset, suffix=suffix) for row in _read_jsonl(path)]
        path.write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
            encoding="utf-8",
        )
    manifest_path = destination / "polymarket_corpus_manifest.json"
    manifest = _read_json(manifest_path)
    manifest["post_freeze_holdout_test_transform"] = {
        "timestamp_offset_ms": offset,
        "id_suffix": suffix,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def _transform_payload(
    value: Any,
    *,
    offset: int,
    suffix: str,
    key: str | None = None,
) -> Any:
    if isinstance(value, dict):
        return {
            child_key: _transform_payload(
                child_value,
                offset=offset,
                suffix=suffix,
                key=child_key,
            )
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [
            _transform_payload(item, offset=offset, suffix=suffix, key=key)
            for item in value
        ]
    if isinstance(value, str) and key in {
        "market_id",
        "condition_id",
        "slug",
        "token_id",
        "up_token_id",
        "down_token_id",
    }:
        return value + suffix
    if isinstance(value, int) and (key == "ts" or str(key).endswith("_ts")):
        return value + offset
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
