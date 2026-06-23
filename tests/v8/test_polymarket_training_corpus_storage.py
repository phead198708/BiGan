"""Training corpus storage tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bigan.v8.polymarket import export_trainable_corpus, round_corpus_id_from_corpus_dir


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


def test_round_corpus_id_from_corpus_dir_uses_single_round_slug(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "phase2_corpus"
    corpus_dir.mkdir()
    (corpus_dir / "polymarket_market_metadata.jsonl").write_text(
        json.dumps({"slug": "btc-updown-5m-1782226500"}) + "\n",
        encoding="utf-8",
    )

    assert round_corpus_id_from_corpus_dir(corpus_dir) == "btc-updown-5m-1782226500"


def test_round_corpus_id_from_corpus_dir_rejects_multi_round_corpus(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "phase2_corpus"
    corpus_dir.mkdir()
    (corpus_dir / "polymarket_market_metadata.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"slug": "btc-updown-5m-1782226500"}),
                json.dumps({"slug": "btc-updown-5m-1782226800"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly one round slug"):
        round_corpus_id_from_corpus_dir(corpus_dir)
