"""Tests for converting historical data/live Polymarket observations to Phase 2 corpus."""

from __future__ import annotations

import json
from pathlib import Path

from bigan.v8.polymarket.contracts import looks_like_sha256
from bigan.v8.polymarket.corpus import (
    RAW_CORPUS_FILENAMES,
    LiveSignalCorpusConversionConfig,
    convert_live_signals_to_phase2_corpus,
)


def test_live_signal_converter_builds_phase2_corpus_from_verified_fixture(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    _write_verified_live_signals(input_dir)

    result = convert_live_signals_to_phase2_corpus(
        LiveSignalCorpusConversionConfig(
            input_path=input_dir,
            output_dir=tmp_path / "out",
        )
    )

    assert result.report["training_eligible"] is True
    assert result.report["phase2_corpus_built"] is True
    assert result.report["accepted_market_count"] == 1
    assert result.report["accepted_resolution_count"] == 1
    assert result.report["rejected_item_count"] == 0
    assert result.report["conversion_policy"]["uses_model_signal_as_label"] is False
    assert result.report["conversion_policy"]["requires_verified_resolution_source"] is True
    assert result.report["paper_only"] is True
    assert result.report["capital_at_risk"] is False
    assert result.phase2_result is not None

    for filename in RAW_CORPUS_FILENAMES:
        assert (result.raw_dir / filename).exists()
    for digest in result.artifact_hashes.values():
        assert looks_like_sha256(digest)

    manifest = _read_json(result.artifact_paths["conversion_manifest"])
    assert manifest["training_eligible"] is True
    assert manifest["phase2_corpus_built"] is True
    for digest in manifest["raw_artifact_hashes"].values():
        assert looks_like_sha256(digest)

    phase2_dir = result.phase2_result.output_dir
    labels = _read_jsonl(phase2_dir / "polymarket_label_rows.jsonl")
    assert any(row["action"] == "BUY_UP_HOLD_TO_SETTLEMENT" for row in labels)
    assert any(row["action"] == "BUY_DOWN_HOLD_TO_SETTLEMENT" for row in labels)
    assert {row["resolved_outcome"] for row in labels} == {"UP"}


def test_live_signal_converter_rejects_unverified_legacy_signal_rows(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_jsonl(
        input_dir / "signals.jsonl",
        [
            {
                "round_slug": "btc-updown-15m-1782000000",
                "round_end_ts": 1_782_000_900_000,
                "ts": 1_782_000_050_000,
                "outcome_side": "UP",
                "selected_side": "UP",
                "token_id": "up-token",
                "opposite_token_id": "down-token",
                "polymarket_price": 0.53,
                "market_implied_prob": 0.53,
                "model_probability": 0.61,
                "edge": 0.08,
            }
        ],
    )

    result = convert_live_signals_to_phase2_corpus(
        LiveSignalCorpusConversionConfig(
            input_path=input_dir,
            output_dir=tmp_path / "out",
        )
    )

    assert result.report["training_eligible"] is False
    assert result.report["phase2_corpus_built"] is False
    assert result.report["accepted_market_count"] == 0
    assert result.report["accepted_orderbook_row_count"] == 0
    assert result.report["accepted_resolution_count"] == 0
    assert result.report["reject_reason_counts"]["missing_executable_up_down_orderbook"] == 1
    assert result.report["reject_reason_counts"]["missing_verified_resolution"] == 1
    assert result.report["reject_reason_counts"]["missing_btc_reference_candles"] == 1
    assert result.report["reject_reason_counts"]["missing_verified_resolution_source"] == 1

    rejected = _read_jsonl(result.artifact_paths["rejected_rows"])
    assert rejected
    assert {
        "missing_executable_up_down_orderbook",
        "missing_verified_resolution",
        "missing_btc_reference_candles",
        "missing_verified_resolution_source",
    }.issubset(set(rejected[-1]["reject_reasons"]))


def test_live_signal_converter_rejects_resolution_prices_without_official_source(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    _write_verified_live_signals(input_dir, include_settlement_reference_source=False)

    result = convert_live_signals_to_phase2_corpus(
        LiveSignalCorpusConversionConfig(
            input_path=input_dir,
            output_dir=tmp_path / "out",
        )
    )

    assert result.report["training_eligible"] is False
    assert result.report["phase2_corpus_built"] is False
    assert result.report["accepted_market_count"] == 0
    assert result.report["reject_reason_counts"]["missing_verified_resolution_source"] == 1
    rejected = _read_jsonl(result.artifact_paths["rejected_rows"])
    assert rejected[-1]["reject_reasons"] == ["missing_verified_resolution_source"]


def test_live_signal_converter_does_not_use_selected_side_as_label(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    _write_verified_live_signals(input_dir, selected_side="DOWN")

    result = convert_live_signals_to_phase2_corpus(
        LiveSignalCorpusConversionConfig(
            input_path=input_dir,
            output_dir=tmp_path / "out",
        )
    )
    assert result.phase2_result is not None

    resolutions = _read_jsonl(
        result.phase2_result.output_dir / "polymarket_resolution_events.jsonl"
    )
    labels = _read_jsonl(result.phase2_result.output_dir / "polymarket_label_rows.jsonl")

    assert result.report["conversion_policy"]["uses_model_signal_as_label"] is False
    assert {row["resolved_outcome"] for row in resolutions} == {"UP"}
    assert {row["resolved_outcome"] for row in labels} == {"UP"}
    up_hold_labels = [row for row in labels if row["action"] == "BUY_UP_HOLD_TO_SETTLEMENT"]
    assert up_hold_labels
    assert all(row["settlement_payout"] == 1.0 for row in up_hold_labels)


def test_live_signal_converter_is_deterministic_for_identical_input(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    _write_verified_live_signals(input_dir)

    first = convert_live_signals_to_phase2_corpus(
        LiveSignalCorpusConversionConfig(
            input_path=input_dir,
            output_dir=tmp_path / "first",
        )
    )
    second = convert_live_signals_to_phase2_corpus(
        LiveSignalCorpusConversionConfig(
            input_path=input_dir,
            output_dir=tmp_path / "second",
        )
    )

    assert first.artifact_hashes["conversion_report"] == second.artifact_hashes["conversion_report"]
    assert (
        first.artifact_hashes["raw_polymarket_markets.jsonl"]
        == second.artifact_hashes["raw_polymarket_markets.jsonl"]
    )
    assert first.phase2_result is not None
    assert second.phase2_result is not None
    assert (
        first.phase2_result.artifact_hashes["corpus_manifest"]
        == second.phase2_result.artifact_hashes["corpus_manifest"]
    )


def _write_verified_live_signals(
    input_dir: Path,
    *,
    selected_side: str = "UP",
    include_settlement_reference_source: bool = True,
) -> Path:
    input_dir.mkdir(parents=True, exist_ok=True)
    start_ts = 1_782_000_000_000
    end_ts = start_ts + 900_000
    slug = f"btc-updown-15m-{start_ts // 1000}"
    rows = []
    for index, decision_ts in enumerate((start_ts, start_ts + 300_000, start_ts + 600_000)):
        up_mid = 0.48 + index * 0.04
        down_mid = 1.0 - up_mid
        for outcome in ("UP", "DOWN"):
            row = {
                "round_slug": slug,
                "market_id": "verified-live-btc15m",
                "round_end_ts": end_ts,
                "ts": decision_ts,
                "outcome_side": outcome,
                "selected_side": selected_side,
                "token_id": "up-token" if outcome == "UP" else "down-token",
                "opposite_token_id": "down-token" if outcome == "UP" else "up-token",
                "up_bid": up_mid - 0.01,
                "up_ask": up_mid + 0.01,
                "down_bid": down_mid - 0.01,
                "down_ask": down_mid + 0.01,
                "up_bid_size": 1000.0 + index,
                "up_ask_size": 900.0 + index,
                "down_bid_size": 950.0 + index,
                "down_ask_size": 875.0 + index,
                "btc_mid_price": 65_000.0 + index * 25.0,
                "reference_price_start": 65_000.0,
                "reference_price_end": 65_050.0,
                "model_probability": 0.2 if selected_side == "DOWN" else 0.8,
            }
            if include_settlement_reference_source:
                row["settlement_reference_source"] = "polymarket_official_btc_usd_reference"
            rows.append(row)
    path = input_dir / "signals.jsonl"
    _write_jsonl(path, rows)
    return path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
