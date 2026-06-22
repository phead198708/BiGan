"""BTC 15m Polymarket adapter tests."""

from __future__ import annotations

import pytest

from bigan.v8.phase0 import MarketData
from bigan.v8.polymarket import (
    POLYMARKET_BTC15M_HORIZON_MS,
    PolymarketAdapterError,
    PolymarketPolicySignal,
    build_polymarket_feature_rows,
    build_polymarket_label_rows,
    build_polymarket_paper_decisions,
    normalize_btc15m_binary_market,
    normalize_token_snapshots,
    synthetic_btc15m_market_payload,
    synthetic_btc_market_rows,
    synthetic_token_snapshot_rows,
)


def test_missing_up_token_is_rejected() -> None:
    payload = synthetic_btc15m_market_payload()
    payload["outcomes"] = [
        {"name": "DOWN", "token_id": "token-down"},
        {"name": "FLAT", "token_id": "token-flat"},
    ]

    with pytest.raises(PolymarketAdapterError, match="missing_up_token"):
        normalize_btc15m_binary_market(payload)


def test_missing_down_token_is_rejected() -> None:
    payload = synthetic_btc15m_market_payload()
    payload["outcomes"] = [
        {"name": "UP", "token_id": "token-up"},
        {"name": "FLAT", "token_id": "token-flat"},
    ]

    with pytest.raises(PolymarketAdapterError, match="missing_down_token"):
        normalize_btc15m_binary_market(payload)


def test_non_binary_market_is_rejected() -> None:
    payload = synthetic_btc15m_market_payload()
    payload["outcomes"].append({"name": "FLAT", "token_id": "token-flat"})

    with pytest.raises(PolymarketAdapterError, match="non_binary_market"):
        normalize_btc15m_binary_market(payload)


def test_non_btc_market_is_rejected_in_btc15m_mode() -> None:
    payload = synthetic_btc15m_market_payload()
    payload["title"] = "Ethereum Up or Down - 15m"
    payload["slug"] = "ethereum-up-or-down"

    with pytest.raises(PolymarketAdapterError, match="non_btc_market"):
        normalize_btc15m_binary_market(payload)


def test_non_15m_market_is_rejected_in_btc15m_mode() -> None:
    payload = synthetic_btc15m_market_payload()
    payload["market_end_ts"] = payload["market_start_ts"] + 5 * 60 * 1000

    with pytest.raises(PolymarketAdapterError, match="non_15m_market"):
        normalize_btc15m_binary_market(payload)


def test_unknown_settlement_rule_is_rejected() -> None:
    payload = synthetic_btc15m_market_payload()
    payload["settlement_rule"] = "UP wins by an unspecified oracle process."

    with pytest.raises(PolymarketAdapterError, match="unknown_settlement_rule"):
        normalize_btc15m_binary_market(payload)


def test_feature_rows_are_v8_compatible_and_causal() -> None:
    market, snapshots, btc_rows = _fixture()

    rows = build_polymarket_feature_rows(
        market=market,
        token_snapshots=snapshots,
        btc_market_data=btc_rows,
    )
    first = rows[0]

    assert rows
    assert first.horizon_ms == POLYMARKET_BTC15M_HORIZON_MS
    assert first.v8_feature.instrument_id == market.slug
    assert "btc_mid_price" in first.features
    assert "up_token_mid_price" in first.features
    assert first.feature_cutoff_ts <= first.decision_ts
    assert first.max_input_ts <= first.decision_ts
    assert all(
        provenance.available_at_ts <= first.decision_ts
        and provenance.input_end_ts <= first.decision_ts
        for provenance in first.v8_feature.provenance.values()
    )


def test_label_rows_include_settlement_metadata_and_net_return() -> None:
    market, snapshots, btc_rows = _fixture()

    labels = build_polymarket_label_rows(
        market=market,
        token_snapshots=snapshots,
        reference_price_end=btc_rows[-1].effective_mid_price,
    )
    by_outcome = {label.outcome: label for label in labels}

    assert set(by_outcome) == {"UP", "DOWN"}
    assert by_outcome["UP"].raw_settlement_metadata_hash
    assert by_outcome["UP"].reference_price_start == market.reference_price_at_start
    assert by_outcome["UP"].reference_price_end > by_outcome["UP"].reference_price_start
    assert by_outcome["UP"].exit_token_price == 1.0
    assert by_outcome["DOWN"].exit_token_price == 0.0
    assert by_outcome["UP"].total_cost == pytest.approx(
        by_outcome["UP"].spread_cost
        + by_outcome["UP"].fee_cost
        + by_outcome["UP"].slippage_cost
        + by_outcome["UP"].liquidity_impact_cost
    )
    assert by_outcome["UP"].net_return == pytest.approx(
        by_outcome["UP"].gross_return - by_outcome["UP"].total_cost
    )
    assert by_outcome["UP"].v8_label.horizon_ms == POLYMARKET_BTC15M_HORIZON_MS


def test_low_confidence_creates_no_trade_decision() -> None:
    market, snapshots, btc_rows = _fixture()
    features = build_polymarket_feature_rows(
        market=market,
        token_snapshots=snapshots,
        btc_market_data=btc_rows,
    )
    signal = PolymarketPolicySignal(
        decision_ts=features[0].decision_ts,
        action=0.5,
        confidence=0.2,
        score=0.1,
        estimated_up_probability=0.8,
    )

    decisions = build_polymarket_paper_decisions(
        market=market,
        feature_rows=(features[0],),
        token_snapshots=snapshots,
        policy_signals=(signal,),
    )

    assert decisions[0].selected_outcome == "NO_TRADE"
    assert "low_confidence" in decisions[0].reason_codes
    assert decisions[0].paper_only is True
    assert decisions[0].capital_at_risk is False


def test_missing_token_price_creates_no_trade_decision() -> None:
    market, snapshots, btc_rows = _fixture()
    features = build_polymarket_feature_rows(
        market=market,
        token_snapshots=snapshots,
        btc_market_data=btc_rows,
    )
    signal = PolymarketPolicySignal(
        decision_ts=features[0].decision_ts,
        action=0.4,
        confidence=0.9,
        score=0.9,
        estimated_up_probability=0.2,
    )
    up_only_snapshots = tuple(
        snapshot
        for snapshot in snapshots
        if snapshot.ts != features[0].decision_ts or snapshot.outcome == "UP"
    )

    decisions = build_polymarket_paper_decisions(
        market=market,
        feature_rows=(features[0],),
        token_snapshots=up_only_snapshots,
        policy_signals=(signal,),
    )

    assert decisions[0].selected_outcome == "NO_TRADE"
    assert "missing_token_price" in decisions[0].reason_codes


def _fixture():
    market = normalize_btc15m_binary_market(synthetic_btc15m_market_payload())
    snapshots = normalize_token_snapshots(
        market=market,
        rows=synthetic_token_snapshot_rows(market),
    )
    btc_rows = tuple(MarketData(**row) for row in synthetic_btc_market_rows(market))
    return market, snapshots, btc_rows
