"""Point-in-time and settlement-semantics tests for Polymarket corpus rows."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.contracts import looks_like_sha256
from bigan.v8.polymarket.corpus import (
    PolymarketCorpusBuildConfig,
    build_polymarket_btc_corpus,
    write_deterministic_polymarket_corpus_fixtures,
)


def test_feature_rows_are_strictly_point_in_time(tmp_path: Path) -> None:
    result = _build_fixture_corpus(tmp_path)
    features = _read_jsonl(result.output_dir / "polymarket_feature_rows.jsonl")

    for row in features:
        assert row["feature_cutoff_ts"] <= row["decision_ts"]
        assert row["max_input_ts"] <= row["decision_ts"]
        assert row["available_at_ts"] <= row["decision_ts"]
        assert set(row["features"]) == set(row["feature_provenance"])
        for provenance in row["feature_provenance"].values():
            assert provenance["input_end_ts"] <= row["decision_ts"]
            assert provenance["available_at_ts"] <= row["decision_ts"]


def test_future_book_snapshot_is_not_used_as_feature_or_entry_label(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    write_deterministic_polymarket_corpus_fixtures(raw_dir)
    original_first_up = _append_future_up_snapshot(raw_dir)

    result = build_polymarket_btc_corpus(
        PolymarketCorpusBuildConfig(
            input_dir=raw_dir,
            output_dir=tmp_path / "corpus",
        )
    )
    features = _read_jsonl(result.output_dir / "polymarket_feature_rows.jsonl")
    labels = _read_jsonl(result.output_dir / "polymarket_label_rows.jsonl")

    first_feature = next(
        row
        for row in features
        if row["market_id"] == "btc5m-up"
        and row["decision_ts"] == original_first_up["ts"]
    )
    assert first_feature["features"]["up_bid"] == pytest.approx(original_first_up["bid_price"])
    assert first_feature["features"]["up_ask"] == pytest.approx(original_first_up["ask_price"])
    assert first_feature["features"]["up_bid"] != pytest.approx(0.97)
    assert first_feature["features"]["up_ask"] != pytest.approx(0.98)

    first_entry_label = next(
        row
        for row in labels
        if row["market_id"] == "btc5m-up"
        and row["decision_ts"] == original_first_up["ts"]
        and row["action"] == "BUY_UP_HOLD_TO_SETTLEMENT"
    )
    assert first_entry_label["entry_bid"] == pytest.approx(original_first_up["bid_price"])
    assert first_entry_label["entry_ask"] == pytest.approx(original_first_up["ask_price"])


def test_current_kline_close_is_not_used_before_candle_is_available(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    write_deterministic_polymarket_corpus_fixtures(raw_dir)
    sentinel = _make_current_open_candle_extreme(raw_dir)

    result = build_polymarket_btc_corpus(
        PolymarketCorpusBuildConfig(
            input_dir=raw_dir,
            output_dir=tmp_path / "corpus",
        )
    )
    features = _read_jsonl(result.output_dir / "polymarket_feature_rows.jsonl")

    open_feature = _feature_at(
        features=features,
        market_id="btc5m-up",
        decision_ts=sentinel["current_open_ts"],
    )
    assert open_feature["features"]["btc_mid_price"] == pytest.approx(
        sentinel["previous_close"]
    )
    assert open_feature["features"]["btc_mid_price"] != pytest.approx(
        sentinel["current_close"]
    )

    closed_feature = _feature_at(
        features=features,
        market_id="btc5m-up",
        decision_ts=sentinel["current_open_ts"] + sentinel["timeframe_ms"],
    )
    assert closed_feature["features"]["btc_mid_price"] == pytest.approx(
        sentinel["current_close"]
    )


def test_train_shadow_split_is_temporal_and_leak_free(tmp_path: Path) -> None:
    result = _build_fixture_corpus(tmp_path)
    split = _read_json(result.output_dir / "polymarket_train_shadow_split.json")

    assert split["train_label_count"] > 0
    assert split["shadow_label_count"] > 0
    assert split["max_train_decision_ts"] < split["split_ts"]
    assert split["min_shadow_decision_ts"] >= split["split_ts"]
    assert split["max_train_decision_ts"] < split["min_shadow_decision_ts"]
    assert looks_like_sha256(split["split_hash"])
    assert looks_like_sha256(split["train_dataset_hash"])
    assert looks_like_sha256(split["shadow_dataset_hash"])
    assert split["paper_only"] is True
    assert split["capital_at_risk"] is False
    assert split["polymarket_write_enabled"] is False
    assert split["wallet_signing_enabled"] is False


def test_tie_and_unknown_resolution_semantics_are_represented_in_labels(
    tmp_path: Path,
) -> None:
    result = _build_fixture_corpus(tmp_path)
    labels = _read_jsonl(result.output_dir / "polymarket_label_rows.jsonl")
    resolutions = {
        row["market_id"]: row
        for row in _read_jsonl(result.output_dir / "polymarket_resolution_events.jsonl")
    }

    tie_resolution = resolutions["btc15m-gte-tie"]
    assert tie_resolution["resolution_status"] == "normal"
    assert tie_resolution["resolved_outcome"] == "UP"
    assert tie_resolution["payout_up"] == 1.0
    assert tie_resolution["payout_down"] == 0.0

    tie_labels = [row for row in labels if row["market_id"] == "btc15m-gte-tie"]
    assert tie_labels
    for row in tie_labels:
        assert row["resolution_status"] == "normal"
        assert row["resolved_outcome"] == "UP"
        assert row["comparator"] == "close_gte_open"
        assert row["tie_breaker"] == "up"

    unknown_resolution = resolutions["btc1h-unknown"]
    assert unknown_resolution["resolution_status"] == "unknown_50_50"
    assert unknown_resolution["resolved_outcome"] == "UNKNOWN_50_50"
    assert unknown_resolution["payout_up"] == 0.5
    assert unknown_resolution["payout_down"] == 0.5

    unknown_hold_labels = [
        row
        for row in labels
        if row["market_id"] == "btc1h-unknown" and row["action"].endswith("HOLD_TO_SETTLEMENT")
    ]
    assert unknown_hold_labels
    for row in unknown_hold_labels:
        assert row["resolution_status"] == "unknown_50_50"
        assert row["resolved_outcome"] == "UNKNOWN_50_50"
        assert row["settlement_payout"] == 0.5
        assert row["tie_breaker"] == "unknown"


def _append_future_up_snapshot(raw_dir: Path) -> dict:
    path = raw_dir / "raw_polymarket_orderbooks.jsonl"
    rows = _read_jsonl(path)
    first_up = next(row for row in rows if row["market_id"] == "btc5m-up" and row["outcome"] == "UP")
    future_up = {
        **first_up,
        "ts": first_up["ts"] + 1_000,
        "available_at_ts": first_up["ts"] + 1_000,
        "bid_price": 0.97,
        "ask_price": 0.98,
        "mid_price": 0.975,
        "bid_size": 10_000.0,
        "ask_size": 10_000.0,
        "liquidity_depth": 20_000.0,
    }
    rows.append(future_up)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    return first_up


def _make_current_open_candle_extreme(raw_dir: Path) -> dict:
    markets = _read_jsonl(raw_dir / "raw_polymarket_markets.jsonl")
    start_ts = next(row["market_start_ts"] for row in markets if row["market_id"] == "btc5m-up")
    path = raw_dir / "raw_binance_btcusdt_klines.jsonl"
    candles = _read_jsonl(path)
    current = next(row for row in candles if row["ts"] == start_ts)
    previous = next(row for row in candles if row["ts"] == start_ts - row["timeframe_ms"])
    current_close = 72_000.0
    current.update(
        {
            "open_price": 65_000.0,
            "high_price": current_close + 10.0,
            "low_price": 64_990.0,
            "close_price": current_close,
            "available_at_ts": current["ts"] + current["timeframe_ms"],
            "close_time": current["ts"] + current["timeframe_ms"],
        }
    )
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in candles),
        encoding="utf-8",
    )
    return {
        "current_open_ts": current["ts"],
        "timeframe_ms": current["timeframe_ms"],
        "previous_close": previous["close_price"],
        "current_close": current_close,
    }


def _feature_at(*, features: list[dict], market_id: str, decision_ts: int) -> dict:
    return next(
        row
        for row in features
        if row["market_id"] == market_id and row["decision_ts"] == decision_ts
    )


def _build_fixture_corpus(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    write_deterministic_polymarket_corpus_fixtures(raw_dir)
    return build_polymarket_btc_corpus(
        PolymarketCorpusBuildConfig(
            input_dir=raw_dir,
            output_dir=tmp_path / "corpus",
        )
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
