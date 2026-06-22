"""Polymarket settlement-engine artifact and PnL tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bigan.v8.phase0 import MarketData
from bigan.v8.polymarket import (
    PolymarketPolicySignal,
    build_polymarket_feature_rows,
    build_polymarket_paper_decisions,
    normalize_btc15m_binary_market,
    normalize_token_snapshots,
    run_polymarket_settlement_engine,
    synthetic_btc15m_market_payload,
    synthetic_btc_market_rows,
    synthetic_token_snapshot_rows,
)


def test_holding_winning_up_token_to_settlement_produces_positive_pnl(
    tmp_path: Path,
) -> None:
    market, snapshots, decisions = _fixture_decisions(outcome="UP")

    result = run_polymarket_settlement_engine(
        market=market,
        token_snapshots=snapshots,
        decisions=decisions,
        reference_price_end=market.reference_price_at_start + 10.0,
        output_dir=tmp_path,
    )
    pnl = result.pnl_breakdown
    settlement = result.settlement_events[0]

    assert settlement.resolved_outcome == "UP"
    assert settlement.qty_up_settled > 0.0
    assert settlement.settlement_pnl > 0.0
    assert pnl["settlement_pnl"] > 0.0
    assert pnl["unresolved_position_count"] == 0
    assert pnl["pnl_reconciled"] is True
    assert _reconciled(pnl)


def test_holding_losing_up_token_to_settlement_produces_negative_pnl(
    tmp_path: Path,
) -> None:
    market, snapshots, decisions = _fixture_decisions(outcome="UP")

    result = run_polymarket_settlement_engine(
        market=market,
        token_snapshots=snapshots,
        decisions=decisions,
        reference_price_end=market.reference_price_at_start - 10.0,
        output_dir=tmp_path,
    )

    assert result.settlement_events[0].resolved_outcome == "DOWN"
    assert result.pnl_breakdown["settlement_pnl"] < 0.0
    assert _reconciled(result.pnl_breakdown)


def test_unknown_50_50_settlement_pays_both_outcomes_half(tmp_path: Path) -> None:
    payload = synthetic_btc15m_market_payload()
    payload["settlement_rule"] = (
        "UP wins if close is greater than open; if unresolved, resolves 50-50."
    )
    market = normalize_btc15m_binary_market(payload)
    snapshots = normalize_token_snapshots(
        market=market,
        rows=synthetic_token_snapshot_rows(market),
    )
    decisions = _decisions_for_market(market, snapshots, outcome="DOWN")

    result = run_polymarket_settlement_engine(
        market=market,
        token_snapshots=snapshots,
        decisions=decisions,
        reference_price_end=market.reference_price_at_start,
        output_dir=tmp_path,
    )

    assert result.settlement_events[0].resolved_outcome == "UNKNOWN_50_50"
    assert result.settlement_events[0].payout_up == 0.5
    assert result.settlement_events[0].payout_down == 0.5


def test_settlement_artifacts_preserve_paper_only_safety_flags(tmp_path: Path) -> None:
    market, snapshots, decisions = _fixture_decisions(outcome="UP")

    result = run_polymarket_settlement_engine(
        market=market,
        token_snapshots=snapshots,
        decisions=decisions,
        reference_price_end=market.reference_price_at_start + 10.0,
        output_dir=tmp_path,
    )

    assert set(result.artifact_paths) == {
        "polymarket_position_ledger",
        "polymarket_settlement_events",
        "polymarket_position_summary",
        "polymarket_pnl_breakdown",
    }
    for name, path in result.artifact_paths.items():
        assert result.artifact_hashes[name]
        rows = _json_or_jsonl(path)
        for row in rows:
            assert row["paper_only"] is True
            assert row["capital_at_risk"] is False
            assert row["polymarket_write_enabled"] is False
            assert row["wallet_signing_enabled"] is False


def test_position_ledger_contains_no_real_order_or_wallet_fields(tmp_path: Path) -> None:
    market, snapshots, decisions = _fixture_decisions(outcome="UP")

    result = run_polymarket_settlement_engine(
        market=market,
        token_snapshots=snapshots,
        decisions=decisions,
        reference_price_end=market.reference_price_at_start + 10.0,
        output_dir=tmp_path,
    )

    forbidden = {"order_id", "wallet_signature", "private_key"}
    for event in result.ledger_events:
        assert forbidden.isdisjoint(event.to_dict())


def _fixture_decisions(outcome: str):
    market = normalize_btc15m_binary_market(synthetic_btc15m_market_payload())
    snapshots = normalize_token_snapshots(
        market=market,
        rows=synthetic_token_snapshot_rows(market),
    )
    decisions = _decisions_for_market(market, snapshots, outcome=outcome)
    return market, snapshots, decisions


def _decisions_for_market(market, snapshots, *, outcome: str):
    btc_rows = tuple(MarketData(**row) for row in synthetic_btc_market_rows(market))
    features = build_polymarket_feature_rows(
        market=market,
        token_snapshots=snapshots,
        btc_market_data=btc_rows,
    )
    signal = PolymarketPolicySignal(
        decision_ts=features[0].decision_ts,
        action=0.8,
        confidence=0.9,
        score=0.9,
        estimated_up_probability=0.8 if outcome == "UP" else 0.2,
    )
    return build_polymarket_paper_decisions(
        market=market,
        feature_rows=(features[0],),
        token_snapshots=snapshots,
        policy_signals=(signal,),
        min_confidence=0.1,
        min_edge=0.0,
        max_paper_size=0.2,
    )


def _json_or_jsonl(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return [json.loads(text)]


def _reconciled(pnl: dict) -> bool:
    total = (
        pnl["realized_trade_pnl"]
        + pnl["settlement_pnl"]
        + pnl["complete_set_pnl"]
        - pnl["fees"]
        - pnl["slippage"]
    )
    return total == pytest.approx(pnl["total_polymarket_pnl"])
