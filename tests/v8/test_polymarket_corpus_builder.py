"""Builder tests for the deterministic Polymarket BTC corpus."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.contracts import looks_like_sha256
from bigan.v8.polymarket.corpus import (
    BTC_UPDOWN_MARKET_HORIZONS_MS,
    NORMALIZED_CORPUS_FILENAMES,
    POLYMARKET_SELL_BEFORE_CLOSE_LABEL_REDESIGN_REPORT_SCHEMA_VERSION,
    POLYMARKET_SELL_BEFORE_CLOSE_LABEL_SCHEMA_VERSION,
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
    label_redesign = _read_json(output_dir / "sell_before_close_label_redesign_report.json")

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
    assert manifest["sell_before_close_label_schema_version"] == (
        POLYMARKET_SELL_BEFORE_CLOSE_LABEL_SCHEMA_VERSION
    )
    assert manifest["sell_before_close_fixed_terminal_bid_only_labels_allowed"] is False
    assert manifest["sell_before_close_label_gate_passed"] is True
    assert summary["sell_before_close_label_gate_passed"] is True
    assert label_redesign["schema_version"] == (
        POLYMARKET_SELL_BEFORE_CLOSE_LABEL_REDESIGN_REPORT_SCHEMA_VERSION
    )
    assert label_redesign["fixed_terminal_bid_only_labels_allowed"] is False
    assert label_redesign["uses_intraround_exit_opportunity_model"] is True
    assert label_redesign["uses_queue_fill_probability_model"] is True
    assert label_redesign["sell_before_close_entry_notional"] == pytest.approx(1.0)
    assert label_redesign["sell_before_close_min_exit_notional"] == pytest.approx(1.0)
    assert label_redesign["min_exit_notional_source"] == "fixed_1_notional"
    assert label_redesign["min_exit_notional_to_entry_notional_ratio"] == pytest.approx(
        1.0
    )
    assert label_redesign["near_miss_threshold"] == pytest.approx(0.95)
    assert label_redesign["near_miss_theoretical_count"] == 0

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
            exit_path = label["sell_before_close_exit_path"]
            assert exit_path["label_source"] == "intraround_executable_exit_path"
            assert exit_path["candidate_exit_snapshot_count"] >= 0
            assert isinstance(exit_path["exit_path_reason_codes"], list)
            assert label["sell_before_close_label_schema_version"] == (
                POLYMARKET_SELL_BEFORE_CLOSE_LABEL_SCHEMA_VERSION
            )
            assert label["sell_before_close_execution_class"] in {
                "realizable_sell_before_close",
                "theoretical_sell_before_close",
                "non_executable_sell_before_close",
            }
            if label["label_uses_executable_exit_path"]:
                assert label["exit_bid"] == pytest.approx(
                    exit_path["best_executable_exit_price"]
                )
                assert label["exit_ask"] == pytest.approx(
                    exit_path["best_executable_exit_ask"]
                )
                assert label["realized_trade_return"] == pytest.approx(
                    label["exit_bid"] / label["entry_ask"] - 1.0
                )
            else:
                assert label["exit_bid"] == 0.0
                assert label["exit_ask"] == 0.0
                assert label["realized_trade_return"] == -1.0
            assert label["settlement_return"] == 0.0
            gross_pnl_per_notional = (
                label["exit_bid"] - label["entry_ask"]
                if label["label_uses_executable_exit_path"]
                else -label["entry_ask"]
            )
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


def test_sell_before_close_label_uses_best_executable_intraround_exit(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    write_deterministic_polymarket_corpus_fixtures(raw_dir)
    orderbooks_path = raw_dir / "raw_polymarket_orderbooks.jsonl"
    rows = _read_jsonl(orderbooks_path)
    for row in rows:
        if row["market_id"] != "btc5m-up" or row["outcome"] != "UP":
            continue
        if row["ts"] == 1_780_100_060_000:
            row["bid_price"] = 0.88
            row["ask_price"] = 0.90
            row["mid_price"] = 0.89
            row["bid_size"] = 1000.0
            row["liquidity_depth"] = 2000.0
        if row["ts"] == 1_780_100_240_000:
            row["bid_price"] = 0.52
            row["ask_price"] = 0.54
            row["mid_price"] = 0.53
            row["bid_size"] = 1000.0
            row["liquidity_depth"] = 2000.0
    _write_jsonl(orderbooks_path, rows)

    result = build_polymarket_btc_corpus(
        PolymarketCorpusBuildConfig(
            input_dir=raw_dir,
            output_dir=tmp_path / "corpus",
        )
    )
    labels = _read_jsonl(result.output_dir / "polymarket_label_rows.jsonl")
    label = next(
        row
        for row in labels
        if row["market_id"] == "btc5m-up"
        and row["decision_ts"] == 1_780_100_000_000
        and row["action"] == "BUY_UP_SELL_BEFORE_CLOSE"
    )

    assert label["sell_before_close_execution_class"] == "realizable_sell_before_close"
    assert label["label_uses_executable_exit_path"] is True
    assert label["exit_bid"] == pytest.approx(0.88)
    assert label["sell_before_close_exit_path"]["best_executable_exit_ts"] == (
        1_780_100_060_000
    )
    assert label["theoretical_terminal_bid_return"] == pytest.approx(
        0.52 / label["entry_ask"] - 1.0
    )


def test_theoretical_terminal_bid_without_executable_liquidity_fails_label_gate(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    write_deterministic_polymarket_corpus_fixtures(raw_dir)
    orderbooks_path = raw_dir / "raw_polymarket_orderbooks.jsonl"
    rows = _read_jsonl(orderbooks_path)
    for row in rows:
        if row["market_id"] == "btc5m-up" and row["outcome"] == "UP":
            if row["ts"] == 1_780_100_000_000:
                row["bid_price"] = 0.30
                row["ask_price"] = 0.32
                row["mid_price"] = 0.31
            else:
                row["bid_price"] = 0.90
                row["ask_price"] = 0.92
                row["mid_price"] = 0.91
            row["bid_size"] = 0.01
            row["liquidity_depth"] = 0.01
    _write_jsonl(orderbooks_path, rows)

    result = build_polymarket_btc_corpus(
        PolymarketCorpusBuildConfig(
            input_dir=raw_dir,
            output_dir=tmp_path / "corpus",
        )
    )
    labels = _read_jsonl(result.output_dir / "polymarket_label_rows.jsonl")
    report = _read_json(result.output_dir / "sell_before_close_label_redesign_report.json")
    label = next(
        row
        for row in labels
        if row["market_id"] == "btc5m-up"
        and row["decision_ts"] == 1_780_100_000_000
        and row["action"] == "BUY_UP_SELL_BEFORE_CLOSE"
    )

    assert label["sell_before_close_execution_class"] == "theoretical_sell_before_close"
    assert label["label_uses_executable_exit_path"] is False
    assert label["exit_bid"] == 0.0
    assert label["realized_trade_return"] == -1.0
    assert label["theoretical_terminal_bid_return"] > 0.0
    exit_path = label["sell_before_close_exit_path"]
    assert exit_path["exit_path_reason_codes"] == [
        "terminal_bid_positive_but_not_executable",
        "min_exit_notional_not_met",
        "queue_fill_probability_below_threshold",
    ]
    assert exit_path["terminal_bid"] == pytest.approx(0.90)
    assert exit_path["best_candidate_bid"] == pytest.approx(0.90)
    assert report["label_gate_passed"] is False
    assert "positive_theoretical_return_without_executable_exit" in report[
        "label_gate_reason_codes"
    ]
    diagnostic = next(
        row
        for row in report["theoretical_sell_before_close_rows"]
        if row["market_id"] == "btc5m-up"
        and row["decision_ts"] == 1_780_100_000_000
        and row["action"] == "BUY_UP_SELL_BEFORE_CLOSE"
    )
    assert diagnostic["slug"] == label["slug"]
    assert diagnostic["outcome"] == "UP"
    assert diagnostic["entry_ask"] == pytest.approx(label["entry_ask"])
    assert diagnostic["terminal_bid"] == pytest.approx(0.90)
    assert diagnostic["theoretical_terminal_bid_return"] == pytest.approx(
        label["theoretical_terminal_bid_return"]
    )
    assert diagnostic["best_candidate_bid"] == pytest.approx(0.90)
    assert diagnostic["queue_fill_probability_estimate"] == pytest.approx(
        label["queue_fill_probability_estimate"]
    )
    assert diagnostic["executable_liquidity_notional"] == pytest.approx(
        label["executable_liquidity_notional"]
    )
    assert diagnostic["min_exit_notional_met"] is False
    assert diagnostic["exit_path_reason_code"] == (
        "terminal_bid_positive_but_not_executable"
    )
    assert diagnostic["exit_path_reason_codes"] == exit_path["exit_path_reason_codes"]


def test_sell_before_close_sizing_policy_can_follow_paper_notional(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    write_deterministic_polymarket_corpus_fixtures(raw_dir)

    result = build_polymarket_btc_corpus(
        PolymarketCorpusBuildConfig(
            input_dir=raw_dir,
            output_dir=tmp_path / "corpus",
            sell_before_close_entry_notional=0.20,
            sell_before_close_min_exit_notional=0.20,
        )
    )
    report = _read_json(result.output_dir / "sell_before_close_label_redesign_report.json")

    assert report["sell_before_close_entry_notional"] == pytest.approx(0.20)
    assert report["sell_before_close_min_exit_notional"] == pytest.approx(0.20)
    assert report["min_exit_notional_source"] == "paper_notional"
    assert report["min_exit_notional_to_entry_notional_ratio"] == pytest.approx(1.0)
    assert report["near_miss_threshold"] == pytest.approx(0.19)


def test_near_miss_theoretical_sell_before_close_count_is_reported(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    write_deterministic_polymarket_corpus_fixtures(raw_dir)
    orderbooks_path = raw_dir / "raw_polymarket_orderbooks.jsonl"
    rows = _read_jsonl(orderbooks_path)
    for row in rows:
        if row["market_id"] == "btc5m-up" and row["outcome"] == "UP":
            if row["ts"] == 1_780_100_000_000:
                row["bid_price"] = 0.30
                row["ask_price"] = 0.32
                row["mid_price"] = 0.31
            else:
                row["bid_price"] = 0.90
                row["ask_price"] = 0.92
                row["mid_price"] = 0.91
            row["bid_size"] = 1.06
            row["liquidity_depth"] = 0.01
    _write_jsonl(orderbooks_path, rows)

    result = build_polymarket_btc_corpus(
        PolymarketCorpusBuildConfig(
            input_dir=raw_dir,
            output_dir=tmp_path / "corpus",
        )
    )
    report = _read_json(result.output_dir / "sell_before_close_label_redesign_report.json")
    diagnostic = next(
        row
        for row in report["theoretical_sell_before_close_rows"]
        if row["market_id"] == "btc5m-up"
        and row["decision_ts"] == 1_780_100_000_000
        and row["action"] == "BUY_UP_SELL_BEFORE_CLOSE"
    )

    assert report["near_miss_threshold"] == pytest.approx(0.95)
    assert report["near_miss_theoretical_count"] >= 1
    assert diagnostic["executable_liquidity_notional"] == pytest.approx(0.954)
    assert diagnostic["min_exit_notional_met"] is False
    assert diagnostic["min_queue_fill_probability_met"] is True
    assert diagnostic["exit_path_reason_codes"] == [
        "terminal_bid_positive_but_not_executable",
        "min_exit_notional_not_met",
    ]


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
