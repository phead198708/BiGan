"""Training corpus storage tests."""

from __future__ import annotations

import json
from pathlib import Path

from bigan.v8.polymarket import export_trainable_corpus


def test_export_trainable_corpus_copies_only_corpus_dir(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "phase2_corpus"
    corpus_dir.mkdir()
    (corpus_dir / "polymarket_corpus_manifest.json").write_text("{}", encoding="utf-8")
    (corpus_dir / "polymarket_feature_rows.jsonl").write_text("[]\n", encoding="utf-8")
    artifact_dir = tmp_path / "recorder_run"
    artifact_dir.mkdir()
    (artifact_dir / "real_corpus_recorder_report.json").write_text("{}", encoding="utf-8")

    target = export_trainable_corpus(
        corpus_dir=corpus_dir,
        corpus_id="corpus-a",
        destination_root=tmp_path / "training_root",
        overwrite_existing=False,
        provenance={"real_historical_corpus_used": True},
    )

    assert target == tmp_path / "training_root" / "polymarket" / "corpus-a"
    assert (target / "polymarket_corpus_manifest.json").exists()
    assert not (target / "real_corpus_recorder_report.json").exists()
    provenance = json.loads((target / "training_corpus_provenance.json").read_text())
    assert provenance["real_historical_corpus_used"] is True
