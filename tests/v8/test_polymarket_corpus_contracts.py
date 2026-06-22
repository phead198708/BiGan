"""Contract tests for deterministic Polymarket BTC corpus artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.contracts import looks_like_sha256
from bigan.v8.polymarket.corpus import (
    BTC_UPDOWN_MARKET_HORIZONS_MS,
    RAW_CORPUS_FILENAMES,
    PolymarketCorpusBuildConfig,
    PolymarketCorpusMarket,
    write_deterministic_polymarket_corpus_fixtures,
)


def test_build_config_defaults_cover_btc_5m_15m_and_1h(tmp_path: Path) -> None:
    config = PolymarketCorpusBuildConfig(
        input_dir=tmp_path / "raw",
        output_dir=tmp_path / "corpus",
    )

    assert set(config.market_families) == set(BTC_UPDOWN_MARKET_HORIZONS_MS)
    assert config.resolved_sample_intervals() == {
        "btc_updown_5m": 60,
        "btc_updown_15m": 300,
        "btc_updown_1h": 900,
    }
    assert config.paper_only is True
    assert config.capital_at_risk is False
    assert config.polymarket_write_enabled is False
    assert config.wallet_signing_enabled is False

    manifest_config = config.to_manifest_dict()
    assert "input_dir" not in manifest_config
    assert "output_dir" not in manifest_config
    assert manifest_config["market_families"] == [
        "btc_updown_5m",
        "btc_updown_15m",
        "btc_updown_1h",
    ]


def test_build_config_rejects_unsupported_family_and_unsafe_flags(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported market families"):
        PolymarketCorpusBuildConfig(
            input_dir=tmp_path / "raw",
            output_dir=tmp_path / "corpus",
            market_families=("eth_updown_5m",),  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="polymarket_write_enabled"):
        PolymarketCorpusBuildConfig(
            input_dir=tmp_path / "raw",
            output_dir=tmp_path / "corpus",
            polymarket_write_enabled=True,
        )

    with pytest.raises(ValueError, match="wallet_signing_enabled"):
        PolymarketCorpusBuildConfig(
            input_dir=tmp_path / "raw",
            output_dir=tmp_path / "corpus",
            wallet_signing_enabled=True,
        )


def test_market_contract_enforces_horizon_token_identity_and_sha() -> None:
    with pytest.raises(ValueError, match="horizon_ms"):
        PolymarketCorpusMarket(
            market_id="m1",
            condition_id="0xcondition",
            slug="btc-5m",
            market_family="btc_updown_5m",
            horizon_ms=900_000,
            market_start_ts=1_000,
            market_end_ts=301_000,
            settlement_ts=301_000,
            up_token_id="up",
            down_token_id="down",
            reference_price_source="binance_btcusdt",
            settlement_rule="UP if close greater than open.",
            raw_market_sha256="a" * 64,
        )

    with pytest.raises(ValueError, match="UP and DOWN token ids"):
        PolymarketCorpusMarket(
            market_id="m1",
            condition_id="0xcondition",
            slug="btc-5m",
            market_family="btc_updown_5m",
            horizon_ms=BTC_UPDOWN_MARKET_HORIZONS_MS["btc_updown_5m"],
            market_start_ts=1_000,
            market_end_ts=301_000,
            settlement_ts=301_000,
            up_token_id="same",
            down_token_id="same",
            reference_price_source="binance_btcusdt",
            settlement_rule="UP if close greater than open.",
            raw_market_sha256="a" * 64,
        )

    with pytest.raises(ValueError, match="raw_market_sha256"):
        PolymarketCorpusMarket(
            market_id="m1",
            condition_id="0xcondition",
            slug="btc-5m",
            market_family="btc_updown_5m",
            horizon_ms=BTC_UPDOWN_MARKET_HORIZONS_MS["btc_updown_5m"],
            market_start_ts=1_000,
            market_end_ts=301_000,
            settlement_ts=301_000,
            up_token_id="up",
            down_token_id="down",
            reference_price_source="binance_btcusdt",
            settlement_rule="UP if close greater than open.",
            raw_market_sha256="not-a-sha",
        )


def test_fixture_writer_emits_canonical_local_raw_files(tmp_path: Path) -> None:
    paths = write_deterministic_polymarket_corpus_fixtures(tmp_path)

    assert set(paths) == set(RAW_CORPUS_FILENAMES)
    for filename in RAW_CORPUS_FILENAMES:
        path = paths[filename]
        assert path.exists()
        assert path.name == filename
        rows = _read_jsonl(path)
        assert rows
        for row in rows:
            if "raw_" not in filename or filename == "raw_binance_btcusdt_klines.jsonl":
                continue
            assert row["paper_only"] is True
            assert row["capital_at_risk"] is False
            assert row["polymarket_write_enabled"] is False
            assert row["wallet_signing_enabled"] is False

    markets = _read_jsonl(paths["raw_polymarket_markets.jsonl"])
    assert {row["market_family"] for row in markets} == set(BTC_UPDOWN_MARKET_HORIZONS_MS)
    for row in markets:
        assert row["up_token_id"] != row["down_token_id"]

    raw_hashes = {filename: path.read_bytes() for filename, path in paths.items()}
    assert all(raw_hashes.values())
    assert looks_like_sha256("0" * 64)


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
