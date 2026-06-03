from __future__ import annotations

from bigan.execution.v6_gate import (
    V6JointGateConfig,
    build_v6_signal_fields,
    evaluate_v6_joint_side,
    evaluate_v6_settlement_side,
    is_v6_model_version,
    v6_payload_from_snapshot,
)


def test_is_v6_model_version() -> None:
    assert is_v6_model_version("xgboost-v6")
    assert is_v6_model_version("xgboost-v6:issue93")
    assert not is_v6_model_version("xgboost-v5")


def test_v6_joint_gate_admits_up_side() -> None:
    payload = {
        "model_version": "xgboost-v6",
        "p_up": 0.90,
        "p_down": 0.05,
        "p_neutral": 0.05,
        "p_vol_up": 0.90,
        "p_vol_down": 0.10,
    }
    config = V6JointGateConfig(
        settlement_threshold=0.50,
        neutral_cap=0.25,
        volatility_threshold=0.50,
        round_trip_cost=0.04,
        ev_margin=0.01,
        gain_priors=(("up", 0.30), ("down", 0.30)),
    )
    assert evaluate_v6_joint_side(payload, config) == "UP"


def test_v6_settlement_gate_ignores_neutral_and_volatility_heads() -> None:
    payload = {
        "model_version": "xgboost-v6",
        "p_up": 0.64,
        "p_down": 0.30,
        "p_neutral": 0.99,
        "p_vol_up": 0.01,
        "p_vol_down": 0.99,
    }
    config = V6JointGateConfig(settlement_threshold=0.50)

    assert evaluate_v6_settlement_side(payload, config) == "UP"
    assert evaluate_v6_joint_side(payload, config) is None


def test_build_v6_signal_fields_maps_down_token() -> None:
    snapshot = {
        "canonical_symbol": "BTC-15M:btc-updown-15m-1000:UP",
        "source_symbol": "token-up",
        "market_implied_prob": 0.40,
        "p_up": 0.10,
        "p_down": 0.85,
        "p_neutral": 0.05,
        "p_vol_up": 0.10,
        "p_vol_down": 0.10,
    }
    config = V6JointGateConfig(
        settlement_threshold=0.50,
        neutral_cap=0.25,
        volatility_threshold=0.50,
        round_trip_cost=0.04,
        ev_margin=0.01,
        gain_priors=(("up", 0.30), ("down", 0.30)),
    )
    fields = build_v6_signal_fields(
        event_id="pred-1",
        ts=1_000,
        created_at=2_000,
        snapshot=snapshot,
        model_version="xgboost-v6",
        config=config,
        round_end_ts=1_000_000,
        opposite_token_id="token-down",
    )
    assert fields is not None
    assert fields["outcome_side"] == "DOWN"
    assert fields["token_id"] == "token-down"
    assert fields["v6_joint_side"] == "DOWN"
    assert fields["p_vol_down"] == 0.10
    assert fields["market_implied_prob"] == 0.60
    assert abs(fields["edge"] - 0.25) < 1e-12
    payload = v6_payload_from_snapshot(snapshot, model_version="xgboost-v6")
    assert payload is not None
    assert evaluate_v6_joint_side(payload, config) is None


def test_build_v6_signal_fields_maps_down_snapshot_token() -> None:
    snapshot = {
        "canonical_symbol": "BTC-15M:btc-updown-15m-1000:DOWN",
        "source_symbol": "token-down",
        "market_implied_prob": 0.40,
        "p_up": 0.10,
        "p_down": 0.85,
        "p_neutral": 0.05,
        "p_vol_up": 0.10,
        "p_vol_down": 0.90,
    }
    config = V6JointGateConfig(
        settlement_threshold=0.50,
        neutral_cap=0.25,
        volatility_threshold=0.50,
        round_trip_cost=0.04,
        ev_margin=0.01,
        gain_priors=(("up", 0.30), ("down", 0.30)),
    )

    fields = build_v6_signal_fields(
        event_id="pred-1",
        ts=1_000,
        created_at=2_000,
        snapshot=snapshot,
        model_version="xgboost-v6",
        config=config,
        round_end_ts=1_000_000,
        opposite_token_id="token-up",
    )

    assert fields is not None
    assert fields["outcome_side"] == "DOWN"
    assert fields["token_id"] == "token-down"
    assert fields["opposite_token_id"] == "token-up"
    assert fields["market_implied_prob"] == 0.40
