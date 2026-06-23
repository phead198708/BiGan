"""Training-gate tests for recorder output passed through Phase 2 corpus builder."""

from __future__ import annotations

import json
from pathlib import Path

from bigan.v8.polymarket.contracts import looks_like_sha256
from bigan.v8.polymarket.recorder import (
    PolymarketRealCorpusRecorderConfig,
    record_polymarket_real_corpus,
)


def test_recorder_output_phase2_corpus_has_temporal_split_and_hashes(tmp_path: Path) -> None:
    result = record_polymarket_real_corpus(
        PolymarketRealCorpusRecorderConfig(
            run_id="training-gate",
            output_dir=tmp_path,
        )
    )
    assert result.phase2_result is not None
    assert result.report["training_eligible"] is True
    assert result.report["phase2_corpus_built"] is True
    assert result.report["fixture_corpus_used"] is False

    split = _read_json(result.phase2_result.output_dir / "polymarket_train_shadow_split.json")
    corpus_manifest = _read_json(
        result.phase2_result.output_dir / "polymarket_corpus_manifest.json"
    )
    recorder_manifest = _read_json(result.artifact_paths["real_corpus_recorder_manifest"])

    assert split["max_train_decision_ts"] < split["min_shadow_decision_ts"]
    assert looks_like_sha256(split["split_hash"])
    assert looks_like_sha256(split["train_dataset_hash"])
    assert looks_like_sha256(split["shadow_dataset_hash"])
    assert corpus_manifest["raw_artifact_hashes"] == recorder_manifest["raw_artifact_hashes"]
    assert looks_like_sha256(recorder_manifest["phase2_corpus_manifest_sha256"])
    assert recorder_manifest["real_historical_corpus_used"] is False
    assert recorder_manifest["fixture_corpus_used"] is False


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
