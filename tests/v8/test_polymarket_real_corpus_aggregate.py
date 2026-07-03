"""Aggregate corpus tests for real Polymarket round corpora."""

from __future__ import annotations

import json
from pathlib import Path

from bigan.v8.polymarket.corpus import (
    SPARSE_THEORETICAL_TRAINING_EXCLUSION_REASON,
    PolymarketCorpusBuildConfig,
    PolymarketRealCorpusAggregateConfig,
    build_polymarket_btc_corpus,
    build_polymarket_real_corpus_aggregate,
    write_deterministic_polymarket_corpus_fixtures,
)


def test_real_corpus_aggregate_excludes_sparse_theoretical_sources(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source_corpora"
    _build_source_corpus(
        source_root=source_root,
        source_id="ok-round",
        suffix="ok",
        sparse_theoretical=False,
    )
    sparse_market_id = _build_source_corpus(
        source_root=source_root,
        source_id="sparse-round",
        suffix="sparse",
        sparse_theoretical=True,
    )

    result = build_polymarket_real_corpus_aggregate(
        PolymarketRealCorpusAggregateConfig(
            source_root=source_root,
            output_dir=tmp_path / "aggregate",
            run_id="aggregate-run",
            market_families=("btc_updown_5m",),
            sell_before_close_entry_notional=0.20,
            sell_before_close_min_exit_notional=0.20,
        )
    )
    report = result.report
    source_manifest = _read_json(result.artifact_paths["aggregate_source_corpora"])
    corpus_manifest = _read_json(result.corpus_dir / "polymarket_corpus_manifest.json")

    assert report["included_source_corpus_count"] == 1
    assert report["excluded_source_corpus_count"] == 1
    assert report["source_corpora_excluded_count"] == 1
    assert report["excluded_market_count"] == 1
    assert report["excluded_market_ids"] == [sparse_market_id]
    assert report["excluded_slugs"] == [f"{sparse_market_id}-fixture"]
    assert report["excluded_reason_counts"] == {
        SPARSE_THEORETICAL_TRAINING_EXCLUSION_REASON: 1
    }
    assert report["sell_before_close_label_gate_passed"] is True
    assert report["theoretical_sell_before_close_count"] == 0
    assert report["sparse_theoretical_sell_before_close_count"] == 0
    assert report["training_eligible"] is True
    assert report["real_historical_training_eligible"] is True
    assert report["manual_live_evidence_eligible"] is True
    assert report["paper_only"] is True
    assert report["capital_at_risk"] is False
    assert report["raw_artifact_hashes"] == corpus_manifest["raw_artifact_hashes"]
    assert source_manifest["excluded"][0]["excluded_reason"] == (
        SPARSE_THEORETICAL_TRAINING_EXCLUSION_REASON
    )


def test_real_corpus_aggregate_can_keep_sparse_sources_for_diagnostics(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source_corpora"
    _build_source_corpus(
        source_root=source_root,
        source_id="sparse-round",
        suffix="sparse",
        sparse_theoretical=True,
    )

    result = build_polymarket_real_corpus_aggregate(
        PolymarketRealCorpusAggregateConfig(
            source_root=source_root,
            output_dir=tmp_path / "aggregate",
            run_id="aggregate-run",
            market_families=("btc_updown_5m",),
            sell_before_close_entry_notional=0.20,
            sell_before_close_min_exit_notional=0.20,
            exclude_sparse_theoretical_sell_before_close=False,
        )
    )

    assert result.report["excluded_source_corpus_count"] == 0
    assert result.report["sell_before_close_label_gate_passed"] is False
    assert result.report["training_eligible"] is False
    assert result.report["sparse_theoretical_sell_before_close_count"] >= 1
    assert "positive_theoretical_return_without_executable_exit" in result.report[
        "sell_before_close_label_gate_reason_codes"
    ]


def _build_source_corpus(
    *,
    source_root: Path,
    source_id: str,
    suffix: str,
    sparse_theoretical: bool,
) -> str:
    raw_dir = source_root.parent / f"{source_id}-raw"
    write_deterministic_polymarket_corpus_fixtures(raw_dir)
    market_id = _rewrite_5m_fixture_market(raw_dir=raw_dir, suffix=suffix)
    if sparse_theoretical:
        _make_up_sell_before_close_path_sparse(raw_dir=raw_dir, market_id=market_id)
    build_polymarket_btc_corpus(
        PolymarketCorpusBuildConfig(
            input_dir=raw_dir,
            output_dir=source_root / source_id,
            market_families=("btc_updown_5m",),
            sell_before_close_entry_notional=0.20,
            sell_before_close_min_exit_notional=0.20,
        )
    )
    return market_id


def _rewrite_5m_fixture_market(*, raw_dir: Path, suffix: str) -> str:
    old_market_id = "btc5m-up"
    new_market_id = f"{old_market_id}-{suffix}"
    token_by_outcome = {
        "UP": f"{new_market_id}-up-token",
        "DOWN": f"{new_market_id}-down-token",
    }
    markets_path = raw_dir / "raw_polymarket_markets.jsonl"
    markets = _read_jsonl(markets_path)
    for row in markets:
        if row["market_id"] != old_market_id:
            continue
        row["market_id"] = new_market_id
        row["condition_id"] = f"0xcondition-{suffix}"
        row["slug"] = f"{new_market_id}-fixture"
        row["up_token_id"] = token_by_outcome["UP"]
        row["down_token_id"] = token_by_outcome["DOWN"]
    _write_jsonl(markets_path, markets)

    for filename in ("raw_polymarket_orderbooks.jsonl", "raw_polymarket_trades.jsonl"):
        path = raw_dir / filename
        rows = _read_jsonl(path)
        for row in rows:
            if row["market_id"] != old_market_id:
                continue
            row["market_id"] = new_market_id
            row["token_id"] = token_by_outcome[row["outcome"]]
        _write_jsonl(path, rows)

    resolutions_path = raw_dir / "raw_polymarket_resolutions.jsonl"
    resolutions = _read_jsonl(resolutions_path)
    for row in resolutions:
        if row["market_id"] != old_market_id:
            continue
        row["market_id"] = new_market_id
        row["raw_resolution_text"] = f"{new_market_id} resolved as normal"
    _write_jsonl(resolutions_path, resolutions)
    return new_market_id


def _make_up_sell_before_close_path_sparse(*, raw_dir: Path, market_id: str) -> None:
    orderbooks_path = raw_dir / "raw_polymarket_orderbooks.jsonl"
    rows = _read_jsonl(orderbooks_path)
    sparse_rows = []
    for row in rows:
        if row["market_id"] == market_id and row["outcome"] == "UP":
            if row["ts"] not in {1_780_100_000_000, 1_780_100_060_000}:
                continue
            if row["ts"] == 1_780_100_000_000:
                row["bid_price"] = 0.30
                row["ask_price"] = 0.32
                row["mid_price"] = 0.31
                row["bid_size"] = 10.0
                row["liquidity_depth"] = 20.0
            else:
                row["bid_price"] = 0.90
                row["ask_price"] = 0.92
                row["mid_price"] = 0.91
                row["bid_size"] = 0.01
                row["liquidity_depth"] = 0.01
        sparse_rows.append(row)
    _write_jsonl(orderbooks_path, sparse_rows)


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
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
