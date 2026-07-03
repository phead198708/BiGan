"""Polymarket live paper settlement path tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import bigan.v8.polymarket.live.operator as live_operator
from bigan.v8.polymarket import PolymarketLivePaperConfig, run_polymarket_live_paper
from bigan.v8.polymarket.live.binance_reference_feed import MockBinanceBTCReferenceFeed
from bigan.v8.polymarket.live.contracts import PolymarketLiveMarket
from bigan.v8.polymarket.live.polymarket_feed import MockPolymarketLiveFeed


def test_resolved_mocked_market_writes_settlement_events(tmp_path: Path) -> None:
    result = run_polymarket_live_paper(
        PolymarketLivePaperConfig(
            run_id="resolved-settlement",
            output_dir=tmp_path,
            settlement_mode="resolved",
        )
    )

    settlement_events = _read_jsonl(result.artifact_paths["polymarket_settlement_events"])
    pnl = result.pnl_breakdown

    assert len(settlement_events) == 3
    assert pnl["resolved_market_count"] == 3
    assert pnl["unresolved_market_count"] == 0
    assert pnl["settled_position_count"] > 0
    assert "settlement_pnl" in pnl
    for row in settlement_events:
        assert row["resolved_outcome"] in {"UP", "DOWN", "UNKNOWN_50_50"}
        _assert_safe(row)


def test_delayed_settlement_leaves_unresolved_positions_awaiting_settlement(
    tmp_path: Path,
) -> None:
    result = run_polymarket_live_paper(
        PolymarketLivePaperConfig(
            run_id="delayed-settlement",
            output_dir=tmp_path,
            settlement_mode="delayed",
        )
    )

    settlement_events = _read_jsonl(result.artifact_paths["polymarket_settlement_events"])
    manifest = result.operator_manifest
    pnl = result.pnl_breakdown

    assert settlement_events == []
    assert pnl["resolved_market_count"] == 0
    assert pnl["unresolved_market_count"] > 0
    assert manifest["operator_status"] == "awaiting_settlement"
    assert manifest["operator_recommendation"] == "await_settlement"
    assert "settlement_pending" in manifest["critical_reason_codes"]
    assert manifest["capital_deployment_allowed"] is False
    assert manifest["live_deployment_allowed"] is False


def test_delayed_real_settlement_wait_finalizes_and_exports_training_corpus(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    def fake_real_live_feed_rows(
        config: PolymarketLivePaperConfig,
        *,
        streaming_writer: Any | None = None,
        on_feed_snapshot: Any | None = None,
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        del streaming_writer, on_feed_snapshot
        polymarket_feed = MockPolymarketLiveFeed(config)
        market_rows = polymarket_feed.market_rows()
        markets = tuple(PolymarketLiveMarket(**row) for row in market_rows)
        return (
            market_rows,
            polymarket_feed.orderbook_rows(markets),
            polymarket_feed.trade_rows(markets),
            MockBinanceBTCReferenceFeed(config).tick_rows(markets),
            [],
        )

    def fake_settlement_wait(
        config: PolymarketLivePaperConfig,
        *,
        market_rows: list[dict[str, Any]],
        candle_rows: list[dict[str, Any]],
        streaming_writer: Any | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        del candle_rows, streaming_writer
        resolved_market_rows = [
            {**row, "status": "resolved", "resolution_available": True}
            for row in market_rows
        ]
        resolved_markets = tuple(
            PolymarketLiveMarket(**row) for row in resolved_market_rows
        )
        return (
            resolved_market_rows,
            MockBinanceBTCReferenceFeed(config).candle_rows(resolved_markets),
            {
                "settlement_wait_enabled": True,
                "settlement_wait_timeout_seconds": 600,
                "settlement_wait_poll_interval_seconds": 15,
                "settlement_wait_poll_count": 2,
                "settlement_wait_elapsed_seconds": 15,
                "settlement_wait_timed_out": False,
                "settlement_wait_resolved_market_count": len(resolved_market_rows),
                "settlement_wait_unresolved_market_count": 0,
                "settlement_wait_error_count": 0,
                "settlement_wait_last_error": None,
            },
        )

    monkeypatch.setattr(live_operator, "load_real_live_feed_rows", fake_real_live_feed_rows)
    monkeypatch.setattr(
        live_operator,
        "wait_for_real_live_settlement_rows",
        fake_settlement_wait,
    )

    training_root = tmp_path / "training-root"
    result = run_polymarket_live_paper(
        PolymarketLivePaperConfig(
            run_id="delayed-settlement-export",
            output_dir=tmp_path,
            mock_live=False,
            market_families=("btc_updown_5m",),
            settlement_mode="delayed",
            export_training_corpus=True,
            training_corpus_root=training_root,
            overwrite_existing=True,
        )
    )
    manifest = result.operator_manifest

    assert manifest["operator_status"] == "completed"
    assert manifest["operator_recommendation"] == "continue_paper_run"
    assert manifest["critical_reason_codes"] == []
    assert manifest["settlement_wait_enabled"] is True
    assert manifest["settlement_wait_poll_count"] == 2
    assert manifest["settlement_wait_timed_out"] is False
    assert manifest["resolved_market_count"] == 1
    assert manifest["unresolved_market_count"] == 0
    assert manifest["settlement_pending_count"] == 0
    assert manifest["settlement_resolved_count"] == 1
    assert manifest["training_raw_round_count"] == 1
    assert manifest["export_training_corpus_enabled"] is True
    assert manifest["exported_training_corpus_count"] == 1
    assert manifest["training_corpus_export_error_count"] == 0
    exported_dir = Path(manifest["exported_training_corpus_dirs"][0])
    assert exported_dir.parent == training_root / "polymarket"
    assert (exported_dir / "polymarket_corpus_manifest.json").exists()
    assert (exported_dir / "training_corpus_provenance.json").exists()


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _assert_safe(payload: dict) -> None:
    assert payload["paper_only"] is True
    assert payload["capital_at_risk"] is False
    assert payload["broker_exchange_write_enabled"] is False
    assert payload["live_exchange_write_enabled"] is False
    assert payload["polymarket_write_enabled"] is False
    assert payload["wallet_signing_enabled"] is False
