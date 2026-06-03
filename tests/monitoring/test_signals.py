"""Champion signal tail helpers."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

from bigan.mlops import connect_mlops_db, initialize_mlops_db
from bigan.monitoring import (
    PositionSignalState,
    PredictionEvent,
    evaluate_position_signal,
    format_signal_row,
    latest_signal_cursor,
    read_recent_signal_rows,
    read_signal_rows_after,
    record_prediction_event,
)


def test_signal_rows_are_derived_from_prediction_events(tmp_path: Path) -> None:
    db_path = tmp_path / "mlops.duckdb"
    conn = connect_mlops_db(db_path)
    initialize_mlops_db(conn)
    _record_event(
        conn,
        event_id="evt-buy",
        created_at=1_000,
        probability=0.80,
        market_implied_prob=0.40,
    )
    _record_event(
        conn,
        event_id="evt-hold",
        created_at=2_000,
        probability=0.60,
        market_implied_prob=0.40,
    )
    _record_event(
        conn,
        event_id="evt-no-market",
        created_at=3_000,
        probability=0.90,
        market_implied_prob=None,
    )
    _record_event(
        conn,
        event_id="evt-down",
        created_at=4_000,
        probability=0.10,
        market_implied_prob=0.40,
        outcome_side="DOWN",
    )
    _record_event(
        conn,
        event_id="evt-down-hold",
        created_at=5_000,
        probability=0.90,
        market_implied_prob=0.40,
        outcome_side="DOWN",
    )
    conn.close()

    rows = read_recent_signal_rows(db_path, model_version="xgboost-v3", limit=5)

    assert [row.signal for row in rows] == [
        "BUY_UP",
        "HOLD",
        "NO_SIGNAL",
        "BUY_DOWN",
        "HOLD",
    ]
    assert rows[0].edge == pytest.approx(0.40)
    assert rows[3].edge == pytest.approx(0.50)
    assert rows[0].source_symbol == "tok-up"
    assert rows[0].outcome_side == "UP"
    assert "BUY_UP" in format_signal_row(rows[0])
    assert latest_signal_cursor(db_path, model_version="xgboost-v3") == (
        5_000,
        "evt-down-hold",
    )

    up_rows = read_recent_signal_rows(
        db_path,
        model_version="xgboost-v3",
        outcome_side="UP",
        limit=4,
    )
    assert [row.event_id for row in up_rows] == ["evt-buy", "evt-hold", "evt-no-market"]
    assert latest_signal_cursor(
        db_path,
        model_version="xgboost-v3",
        outcome_side="UP",
    ) == (3_000, "evt-no-market")

    newer = read_signal_rows_after(
        db_path,
        model_version="xgboost-v3",
        after_created_at=1_000,
        after_event_id="evt-buy",
        outcome_side="UP",
    )
    assert [row.event_id for row in newer] == ["evt-hold", "evt-no-market"]


def test_signal_rows_ignore_post_expiry_degenerate_market_prices(tmp_path: Path) -> None:
    db_path = tmp_path / "mlops.duckdb"
    conn = connect_mlops_db(db_path)
    initialize_mlops_db(conn)
    round_start = 1_779_755_400_000
    round_end = round_start + 15 * 60_000
    _record_event(
        conn,
        event_id="evt-post-expiry-dirty",
        created_at=1_000,
        probability=0.99,
        market_implied_prob=1.0,
        canonical_symbol="BTC-15M:btc-updown-15m-1779755400:UP",
        ts=round_end + 60_000,
        features={
            "market_implied_prob": 1.0,
            "spread": 1.0,
            "tick_spread": 1.0,
            "liquidity_bucket": 0.0,
        },
    )
    conn.close()

    rows = read_recent_signal_rows(db_path, model_version="xgboost-v3", limit=1)

    assert rows[0].event_id == "evt-post-expiry-dirty"
    assert rows[0].market_implied_prob is None
    assert rows[0].edge is None
    assert rows[0].signal == "NO_SIGNAL"


def test_signal_rows_retry_transient_read_only_db_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "mlops.duckdb"
    conn = connect_mlops_db(db_path)
    initialize_mlops_db(conn)
    _record_event(
        conn,
        event_id="evt-buy",
        created_at=1_000,
        probability=0.80,
        market_implied_prob=0.40,
    )
    conn.close()

    real_connect = duckdb.connect
    read_only_calls: list[dict[str, object]] = []
    monkeypatch.setenv("BIGAN_MLOPS_CONNECT_RETRY_DELAY_SECONDS", "0")

    def flaky_connect(path: str, *args: object, **kwargs: object):
        if path == str(db_path) and kwargs.get("read_only") is True:
            read_only_calls.append(dict(kwargs))
            if len(read_only_calls) == 1:
                raise duckdb.IOException("IO Error: Could not set lock on file mlops.duckdb")
        return real_connect(path, *args, **kwargs)

    monkeypatch.setattr(duckdb, "connect", flaky_connect)

    rows = read_recent_signal_rows(db_path, model_version="xgboost-v3", limit=5)

    assert len(read_only_calls) == 2
    assert [row.event_id for row in rows] == ["evt-buy"]


def test_position_signal_state_machine_requires_position_before_sell() -> None:
    state = PositionSignalState()

    low_edge_flat = evaluate_position_signal(
        state,
        edge=0.01,
        market_implied_prob=0.55,
        outcome_side="UP",
        event_id="flat-low-edge",
        exit_edge_threshold=0.05,
    )
    assert low_edge_flat.signal == "HOLD"
    assert not low_edge_flat.state.position_open

    buy_down = evaluate_position_signal(
        low_edge_flat.state,
        edge=0.35,
        market_implied_prob=0.45,
        outcome_side="DOWN",
        event_id="buy-down",
    )
    assert buy_down.signal == "BUY_DOWN"
    assert buy_down.state.position_open
    assert buy_down.state.outcome_side == "DOWN"
    assert buy_down.state.entry_event_id == "buy-down"

    hold_open = evaluate_position_signal(
        buy_down.state,
        edge=0.20,
        market_implied_prob=0.50,
        outcome_side="DOWN",
        event_id="still-open",
        exit_edge_threshold=0.05,
    )
    assert hold_open.signal == "HOLD"
    assert hold_open.state == buy_down.state

    opposite_side = evaluate_position_signal(
        hold_open.state,
        edge=-0.45,
        market_implied_prob=0.51,
        outcome_side="UP",
        event_id="opposite-side-low-edge",
        exit_edge_threshold=0.05,
    )
    assert opposite_side.signal == "HOLD"
    assert opposite_side.state == buy_down.state

    sell = evaluate_position_signal(
        opposite_side.state,
        edge=0.02,
        market_implied_prob=0.53,
        outcome_side="DOWN",
        event_id="sell-down",
        exit_edge_threshold=0.05,
    )
    assert sell.signal == "SELL"
    assert sell.reason == "edge_reversal"
    assert not sell.state.position_open


def test_position_signal_state_machine_sells_on_profit_target_and_round_end() -> None:
    open_down = PositionSignalState(
        position_open=True,
        outcome_side="DOWN",
        entry_price=0.40,
        entry_event_id="entry-down",
    )

    profit_target = evaluate_position_signal(
        open_down,
        edge=0.20,
        market_implied_prob=0.56,
        outcome_side="DOWN",
        event_id="take-profit",
    )
    assert profit_target.signal == "SELL"
    assert profit_target.reason == "profit_target"
    assert not profit_target.state.position_open

    near_end = evaluate_position_signal(
        open_down,
        edge=0.20,
        market_implied_prob=0.43,
        outcome_side="DOWN",
        event_id="lock-round-profit",
        current_ts=1_779_840_850_000,
        round_end_ts=1_779_840_900_000,
    )
    assert near_end.signal == "SELL"
    assert near_end.reason == "round_end_profit"
    assert not near_end.state.position_open

    loss_near_end = evaluate_position_signal(
        open_down,
        edge=0.20,
        market_implied_prob=0.39,
        outcome_side="DOWN",
        event_id="loss-near-end",
        current_ts=1_779_840_850_000,
        round_end_ts=1_779_840_900_000,
    )
    assert loss_near_end.signal == "HOLD"
    assert loss_near_end.reason == "open_position"
    assert loss_near_end.state == open_down


def _record_event(
    conn,
    *,
    event_id: str,
    created_at: int,
    probability: float,
    market_implied_prob: float | None,
    outcome_side: str = "UP",
    canonical_symbol: str | None = None,
    ts: int | None = None,
    features: dict[str, float | None] | None = None,
) -> None:
    feature_values = {"market_implied_prob": market_implied_prob}
    if features:
        feature_values.update(features)
    record_prediction_event(
        conn,
        PredictionEvent(
            event_id=event_id,
            ts=1_779_200_000_000 + created_at if ts is None else ts,
            model_version="xgboost-v3",
            feature_version="bigan-mvp-v1.0.0",
            prob_up_15m=probability,
            confidence_bucket="high_up",
            top_features_json="[]",
            feature_hash=f"hash-{event_id}",
            feature_snapshot_json=json.dumps(
                {
                    "source_symbol": "tok-up",
                    "canonical_symbol": canonical_symbol or f"BTC-15M:test:{outcome_side}",
                    "market_implied_prob": market_implied_prob,
                    "features": feature_values,
                }
            ),
            serving_latency_ms=1.0,
            created_at=created_at,
        ),
    )
