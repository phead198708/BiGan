"""Builder tests for the deterministic Polymarket BTC corpus."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.contracts import looks_like_sha256
from bigan.v8.polymarket.corpus import (
    BTC_UPDOWN_MARKET_HORIZONS_MS,
    NORMALIZED_CORPUS_FILENAMES,
    RAW_CORPUS_FILENAMES,
    PolymarketCorpusBuildConfig,
    build_polymarket_btc_corpus,
    write_deterministic_polymarket_corpus_fixtures,
)

EXPECTED_ACTIONS = {
    "NO_TRADE",
    "BUY_UP_HOLD_TO_SETTLEMENT",
    "BUY_DOWN_HOLD_TO_SETTLEMENT",
    "BUY_UP_SELL_BEFORE_CLOSE",
    "BUY_DOWN_SELL_BEFORE_CLOSE",
}


def test_builder_accepts_fixture_files_and_writes_required_outputs(tmp_path: Path) -> None:
    result = _build_fixture_corpus(tmp_path)
    output_dir = result.output_dir

    for filename in NORMALIZED_CORPUS_FILENAMES:
        assert (output_dir / filename).exists(), filename
    assert (output_dir / "polymarket_corpus_manifest.json").exists()

    manifest = _read_json(output_dir / "polymarket_corpus_manifest.json")
    summary = _read_json(output_dir / "polymarket_corpus_summary.json")

    assert manifest["market_count"] == 3
    assert manifest["feature_row_count"] == 12
    assert manifest["label_row_count"] == 60
    assert manifest["market_family_counts"] == {
        "btc_updown_5m": 1,
        "btc_updown_15m": 1,
        "btc_updown_1h": 1,
    }
    assert summary["market_family_counts"] == manifest["market_family_counts"]
    assert summary["split"]["max_train_decision_ts"] < summary["split"]["min_shadow_decision_ts"]
    assert summary["raw_artifact_hashes"] == manifest["raw_artifact_hashes"]
    manifest_normalized_hashes = dict(manifest["normalized_artifact_hashes"])
    manifest_normalized_hashes.pop("corpus_summary")
    assert summary["normalized_artifact_hashes"] == manifest_normalized_hashes
    assert manifest["paper_only"] is True
    assert manifest["capital_at_risk"] is False
    assert manifest["polymarket_write_enabled"] is False
    assert manifest["wallet_signing_enabled"] is False

    assert set(manifest["raw_artifact_hashes"]) == set(RAW_CORPUS_FILENAMES)
    for digest in manifest["raw_artifact_hashes"].values():
        assert looks_like_sha256(digest)
    for digest in manifest["normalized_artifact_hashes"].values():
        assert looks_like_sha256(digest)
    for digest in result.artifact_hashes.values():
        assert looks_like_sha256(digest)


def test_every_market_has_rule_manifest_and_up_down_tokens(tmp_path: Path) -> None:
    result = _build_fixture_corpus(tmp_path)
    markets = _read_jsonl(result.output_dir / "polymarket_market_metadata.jsonl")
    rules = _read_jsonl(result.output_dir / "polymarket_market_rules.jsonl")

    assert {row["market_family"] for row in markets} == set(BTC_UPDOWN_MARKET_HORIZONS_MS)
    rule_by_market = {row["market_id"]: row for row in rules}
    for market in markets:
        assert market["market_id"] in rule_by_market
        assert market["up_token_id"]
        assert market["down_token_id"]
        assert market["up_token_id"] != market["down_token_id"]
        rule = rule_by_market[market["market_id"]]
        assert rule["paper_only"] is True
        assert rule["capital_at_risk"] is False
        assert rule["candle_close_ts"] - rule["candle_open_ts"] == market["horizon_ms"]
        assert looks_like_sha256(rule["raw_rule_sha256"])


def test_labels_use_ask_for_entries_and_bid_for_sell_before_close(
    tmp_path: Path,
) -> None:
    result = _build_fixture_corpus(tmp_path)
    labels = _read_jsonl(result.output_dir / "polymarket_label_rows.jsonl")
    snapshots = _read_jsonl(result.output_dir / "polymarket_token_book_snapshots.jsonl")
    markets = {
        row["market_id"]: row
        for row in _read_jsonl(result.output_dir / "polymarket_market_metadata.jsonl")
    }

    assert {row["action"] for row in labels} == EXPECTED_ACTIONS
    for label in labels:
        if label["action"] == "NO_TRADE":
            assert label["entry_bid"] == 0.0
            assert label["entry_ask"] == 0.0
            assert label["total_net_return"] == 0.0
            assert label["total_net_pnl_per_notional"] == 0.0
            continue

        entry_snapshot = _last_snapshot(
            snapshots=snapshots,
            market_id=label["market_id"],
            outcome=label["outcome"],
            decision_ts=label["decision_ts"],
        )
        assert label["entry_bid"] == pytest.approx(entry_snapshot["bid_price"])
        assert label["entry_ask"] == pytest.approx(entry_snapshot["ask_price"])
        assert label["entry_mid"] == pytest.approx(entry_snapshot["mid_price"])

        if label["action"].endswith("SELL_BEFORE_CLOSE"):
            exit_snapshot = _last_snapshot(
                snapshots=snapshots,
                market_id=label["market_id"],
                outcome=label["outcome"],
                decision_ts=markets[label["market_id"]]["market_end_ts"] - 1,
            )
            assert label["exit_bid"] == pytest.approx(exit_snapshot["bid_price"])
            assert label["exit_ask"] == pytest.approx(exit_snapshot["ask_price"])
            assert label["realized_trade_return"] == pytest.approx(
                label["exit_bid"] / label["entry_ask"] - 1.0
            )
            assert label["settlement_return"] == 0.0
            gross_pnl_per_notional = label["exit_bid"] - label["entry_ask"]
        else:
            assert label["exit_bid"] == 0.0
            assert label["exit_ask"] == 0.0
            assert label["settlement_return"] == pytest.approx(
                label["settlement_payout"] / label["entry_ask"] - 1.0
            )
            gross_pnl_per_notional = label["settlement_payout"] - label["entry_ask"]

        expected_net = (
            label["realized_trade_return"]
            + label["settlement_return"]
            - label["fees"]
            - label["slippage"]
            - label["liquidity_impact"]
        )
        expected_pnl_per_notional = (
            gross_pnl_per_notional
            - label["fees"]
            - label["slippage"]
            - label["liquidity_impact"]
        )
        assert label["total_net_return"] == pytest.approx(expected_net)
        assert label["total_net_pnl_per_notional"] == pytest.approx(
            expected_pnl_per_notional
        )
        assert label["is_positive"] is (label["total_net_return"] > 0.0)


def test_rebuilding_from_identical_fixtures_produces_identical_hashes(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    write_deterministic_polymarket_corpus_fixtures(raw_dir)

    first = build_polymarket_btc_corpus(
        PolymarketCorpusBuildConfig(
            input_dir=raw_dir,
            output_dir=tmp_path / "first",
        )
    )
    second = build_polymarket_btc_corpus(
        PolymarketCorpusBuildConfig(
            input_dir=raw_dir,
            output_dir=tmp_path / "second",
        )
    )

    for artifact_name in (
        "market_rules",
        "market_metadata",
        "token_book_snapshots",
        "token_trades",
        "btc_reference_candles",
        "resolution_events",
        "feature_rows",
        "label_rows",
        "train_shadow_split",
        "corpus_summary",
        "corpus_manifest",
    ):
        assert first.artifact_hashes[artifact_name] == second.artifact_hashes[artifact_name]


def test_builder_accepts_verified_payout_only_resolution_rows(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    write_deterministic_polymarket_corpus_fixtures(raw_dir)
    resolutions_path = raw_dir / "raw_polymarket_resolutions.jsonl"
    resolutions = _read_jsonl(resolutions_path)
    for row in resolutions:
        if row["market_id"] != "btc5m-up":
            continue
        row.pop("reference_price_start")
        row.pop("reference_price_end")
        row["resolution_status"] = "normal"
        row["resolved_outcome"] = "UP"
        row["payout_up"] = 1.0
        row["payout_down"] = 0.0
        row["resolution_source_type"] = "gamma_outcome_prices"
    _write_jsonl(resolutions_path, resolutions)

    result = build_polymarket_btc_corpus(
        PolymarketCorpusBuildConfig(
            input_dir=raw_dir,
            output_dir=tmp_path / "corpus",
        )
    )

    normalized_resolutions = _read_jsonl(
        result.output_dir / "polymarket_resolution_events.jsonl"
    )
    payout_only = next(row for row in normalized_resolutions if row["market_id"] == "btc5m-up")
    assert payout_only["reference_price_start"] is None
    assert payout_only["reference_price_end"] is None
    assert payout_only["resolved_outcome"] == "UP"
    assert payout_only["payout_up"] == 1.0
    labels = _read_jsonl(result.output_dir / "polymarket_label_rows.jsonl")
    settlement_labels = [
        row
        for row in labels
        if row["market_id"] == "btc5m-up" and row["action"] == "BUY_UP_HOLD_TO_SETTLEMENT"
    ]
    assert settlement_labels
    assert {row["settlement_payout"] for row in settlement_labels} == {1.0}


@pytest.mark.parametrize(
    ("filename", "market_id", "outcome", "patch", "error_match"),
    (
        (
            "raw_polymarket_orderbooks.jsonl",
            "btc5m-up",
            "UP",
            {"token_id": "wrong-token", "outcome": "UP"},
            "unknown token_id",
        ),
        (
            "raw_polymarket_orderbooks.jsonl",
            "btc5m-up",
            "UP",
            {"outcome": "DOWN"},
            "token_id/outcome mismatch",
        ),
        (
            "raw_polymarket_trades.jsonl",
            "btc5m-up",
            "DOWN",
            {"token_id": "wrong-token", "outcome": "DOWN"},
            "unknown token_id",
        ),
        (
            "raw_polymarket_trades.jsonl",
            "btc5m-up",
            "DOWN",
            {"outcome": "UP"},
            "token_id/outcome mismatch",
        ),
    ),
)
def test_token_id_outcome_mismatches_fail_closed(
    tmp_path: Path,
    filename: str,
    market_id: str,
    outcome: str,
    patch: dict,
    error_match: str,
) -> None:
    raw_dir = tmp_path / "raw"
    write_deterministic_polymarket_corpus_fixtures(raw_dir)
    _patch_first_row(
        raw_dir / filename,
        market_id=market_id,
        outcome=outcome,
        patch=patch,
    )

    with pytest.raises(ValueError, match=error_match):
        build_polymarket_btc_corpus(
            PolymarketCorpusBuildConfig(
                input_dir=raw_dir,
                output_dir=tmp_path / "corpus",
            )
        )


@pytest.mark.parametrize(
    ("filename", "output_filename"),
    (
        ("raw_polymarket_orderbooks.jsonl", "polymarket_token_book_snapshots.jsonl"),
        ("raw_polymarket_trades.jsonl", "polymarket_token_trades.jsonl"),
    ),
)
def test_missing_token_id_with_valid_outcome_uses_canonical_market_token(
    tmp_path: Path,
    filename: str,
    output_filename: str,
) -> None:
    raw_dir = tmp_path / "raw"
    write_deterministic_polymarket_corpus_fixtures(raw_dir)
    patched = _patch_first_row(
        raw_dir / filename,
        market_id="btc5m-up",
        outcome="UP",
        patch={"token_id": None, "outcome": "UP"},
    )

    result = build_polymarket_btc_corpus(
        PolymarketCorpusBuildConfig(
            input_dir=raw_dir,
            output_dir=tmp_path / "corpus",
        )
    )
    normalized_rows = _read_jsonl(result.output_dir / output_filename)
    canonical_row = next(
        row
        for row in normalized_rows
        if row["market_id"] == patched["market_id"]
        and row["outcome"] == "UP"
        and row["ts"] == patched["ts"]
    )

    assert canonical_row["token_id"] == "btc5m-up-up-token"


def _build_fixture_corpus(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    write_deterministic_polymarket_corpus_fixtures(raw_dir)
    return build_polymarket_btc_corpus(
        PolymarketCorpusBuildConfig(
            input_dir=raw_dir,
            output_dir=tmp_path / "corpus",
        )
    )


def _patch_first_row(
    path: Path,
    *,
    market_id: str,
    outcome: str,
    patch: dict,
) -> dict:
    rows = _read_jsonl(path)
    for row in rows:
        if row["market_id"] == market_id and row["outcome"] == outcome:
            for key, value in patch.items():
                if value is None:
                    row.pop(key, None)
                else:
                    row[key] = value
            _write_jsonl(path, rows)
            return row
    raise AssertionError(f"missing fixture row for {market_id} {outcome}")


def _last_snapshot(
    *,
    snapshots: list[dict],
    market_id: str,
    outcome: str,
    decision_ts: int,
) -> dict:
    eligible = [
        row
        for row in snapshots
        if row["market_id"] == market_id
        and row["outcome"] == outcome
        and row["ts"] <= decision_ts
        and row["available_at_ts"] <= decision_ts
    ]
    assert eligible
    return max(eligible, key=lambda row: (row["ts"], row["available_at_ts"]))


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
