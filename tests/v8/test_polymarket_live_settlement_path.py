"""Polymarket live paper settlement path tests."""

from __future__ import annotations

import json
from pathlib import Path

from bigan.v8.polymarket import PolymarketLivePaperConfig, run_polymarket_live_paper


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


def test_delayed_settlement_leaves_unresolved_positions_and_blocks(
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
    assert manifest["operator_status"] == "blocked_fail_closed"
    assert manifest["operator_recommendation"] == "blocked_fail_closed"
    assert "settlement_pending" in manifest["critical_reason_codes"]
    assert manifest["capital_deployment_allowed"] is False
    assert manifest["live_deployment_allowed"] is False


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
