"""Live dashboard aggregation tests for issue #63."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

from bigan.mlops import connect_mlops_db, initialize_mlops_db
from bigan.monitoring import (
    PredictionEvent,
    PredictionOutcome,
    compute_brier_component,
    read_dashboard_snapshot,
    record_prediction_event,
    record_prediction_outcome,
    render_dashboard,
)


def test_dashboard_snapshot_tracks_current_round_and_settled_pnl(tmp_path: Path) -> None:
    db_path = tmp_path / "mlops.duckdb"
    conn = connect_mlops_db(db_path)
    initialize_mlops_db(conn)

    _record_event(conn, event_id="evt-a-hold", ts=1_000_000, prob=0.45, market=0.40, round_id="round-a")
    _record_event(conn, event_id="evt-a-buy", ts=1_060_000, prob=0.80, market=0.40, round_id="round-a")
    _record_outcome(conn, event_id="evt-a-buy", prob=0.80, label=True, realized_return=0.60)
    _record_event(conn, event_id="evt-b-buy", ts=1_120_000, prob=0.90, market=0.50, round_id="round-b")
    _record_outcome(conn, event_id="evt-b-buy", prob=0.90, label=False, realized_return=-0.50)
    _record_event(conn, event_id="evt-c-buy", ts=1_780_000, prob=0.85, market=0.50, round_id="round-c")
    conn.close()

    snapshot = read_dashboard_snapshot(
        db_path,
        model_version="xgboost-v3",
        edge_threshold=0.30,
        now_ms=1_800_000,
    )

    assert snapshot.current_round is not None
    assert snapshot.current_round.round_id == "BTC-15M:round-c"
    assert snapshot.current_round.latest_signal.signal_type == "BUY_UP"
    assert snapshot.current_round.entry_price == pytest.approx(0.50)
    assert snapshot.session_trade_count == 2
    assert snapshot.session_win_count == 1
    assert snapshot.session_loss_count == 1
    assert snapshot.session_total_pnl == pytest.approx(0.10)
    assert snapshot.edge_trigger_rate_1h is not None

    rendered = render_dashboard(snapshot)
    assert "LIVE CHAMPION DASHBOARD" in rendered
    assert "CURRENT ROUND" in rendered
    assert "ROUND HISTORY" in rendered
    assert "PnL=+0.100" in rendered


def test_dashboard_renders_v7_pm_monitoring_from_phase4_summary(tmp_path: Path) -> None:
    db_path = tmp_path / "mlops.duckdb"
    summary_path = tmp_path / "phase4-summary.json"
    conn = connect_mlops_db(db_path)
    initialize_mlops_db(conn)
    _record_event(conn, event_id="evt-buy", ts=1_000_000, prob=0.80, market=0.40, round_id="round-a")
    conn.close()
    summary_path.write_text(
        json.dumps(
            {
                "v7_pm_monitoring": {
                    "divergence_reduce_hold_edge": {
                        "count": 3,
                        "p50": 0.12,
                        "p90": 0.24,
                    },
                    "take_profit_candidates": {
                        "evaluations": 4,
                        "exits": 1,
                        "unexecuted": 3,
                        "reason_counts": {
                            "convergence_force_exit_before_expiry": 4,
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    snapshot = read_dashboard_snapshot(
        db_path,
        model_version="xgboost-v3",
        edge_threshold=0.30,
        now_ms=1_100_000,
        phase4_summary_path=summary_path,
    )
    rendered = render_dashboard(snapshot)

    assert snapshot.v7_pm_monitoring is not None
    assert "V7 PM MONITORING" in rendered
    assert "count=3 p50=0.120 p90=0.240" in rendered
    assert "unexecuted=3" in rendered


def test_dashboard_snapshot_retries_transient_read_only_db_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "mlops.duckdb"
    conn = connect_mlops_db(db_path)
    initialize_mlops_db(conn)
    _record_event(conn, event_id="evt-buy", ts=1_000_000, prob=0.80, market=0.40, round_id="round-a")
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

    snapshot = read_dashboard_snapshot(
        db_path,
        model_version="xgboost-v3",
        edge_threshold=0.30,
        now_ms=1_100_000,
    )

    assert len(read_only_calls) == 2
    assert snapshot.current_round is not None
    assert snapshot.current_round.latest_signal.event_id == "evt-buy"


def test_dashboard_supports_paper_sell_when_edge_fades(tmp_path: Path) -> None:
    db_path = tmp_path / "mlops.duckdb"
    conn = connect_mlops_db(db_path)
    initialize_mlops_db(conn)
    _record_event(conn, event_id="evt-buy", ts=1_000_000, prob=0.90, market=0.50, round_id="round-sell")
    _record_event(conn, event_id="evt-sell", ts=1_060_000, prob=0.52, market=0.61, round_id="round-sell")
    conn.close()

    snapshot = read_dashboard_snapshot(
        db_path,
        model_version="xgboost-v3",
        edge_threshold=0.30,
        exit_edge_threshold=0.05,
        now_ms=1_100_000,
    )

    assert snapshot.current_round is not None
    assert snapshot.current_round.latest_signal.signal_type == "SELL"
    assert snapshot.current_round.settlement is not None
    assert snapshot.current_round.settlement.realized_pnl == pytest.approx(0.11)


def test_dashboard_does_not_sell_on_post_expiry_degenerate_quote(tmp_path: Path) -> None:
    db_path = tmp_path / "mlops.duckdb"
    conn = connect_mlops_db(db_path)
    initialize_mlops_db(conn)
    round_slug = "btc-updown-15m-1779755400"
    round_start = 1_779_755_400_000
    round_end = round_start + 15 * 60_000
    _record_event(
        conn,
        event_id="evt-buy",
        ts=round_start + 8 * 60_000,
        prob=0.90,
        market=0.50,
        round_id=round_slug,
    )
    _record_event(
        conn,
        event_id="evt-post-expiry-dirty",
        ts=round_end + 60_000,
        prob=0.99,
        market=1.0,
        round_id=round_slug,
        features={
            "market_implied_prob": 1.0,
            "spread": 1.0,
            "tick_spread": 1.0,
            "liquidity_bucket": 0.0,
        },
    )
    conn.close()

    snapshot = read_dashboard_snapshot(
        db_path,
        model_version="xgboost-v3",
        edge_threshold=0.30,
        exit_edge_threshold=0.05,
        now_ms=round_end + 120_000,
    )

    assert snapshot.current_round is not None
    assert [signal.signal_type for signal in snapshot.current_round.signals] == ["BUY_UP", "HOLD"]
    assert snapshot.current_round.signals[-1].market_implied_prob is None
    assert snapshot.current_round.exit_price is None
    assert snapshot.current_round.settlement is None


def test_dashboard_does_not_sell_position_on_opposite_side_row(tmp_path: Path) -> None:
    db_path = tmp_path / "mlops.duckdb"
    conn = connect_mlops_db(db_path)
    initialize_mlops_db(conn)
    _record_event(conn, event_id="evt-buy-up", ts=1_000_000, prob=0.90, market=0.51, round_id="round-mixed")
    _record_event(
        conn,
        event_id="evt-down-low-edge",
        ts=1_030_000,
        prob=0.99,
        market=0.50,
        round_id="round-mixed",
        outcome_side="DOWN",
    )
    _record_event(conn, event_id="evt-sell-up", ts=1_060_000, prob=0.52, market=0.61, round_id="round-mixed")
    conn.close()

    snapshot = read_dashboard_snapshot(
        db_path,
        model_version="xgboost-v3",
        edge_threshold=0.30,
        exit_edge_threshold=0.05,
        outcome_side="ANY",
        now_ms=1_100_000,
    )

    assert snapshot.current_round is not None
    signals = snapshot.current_round.signals
    assert [signal.signal_type for signal in signals] == ["BUY_UP", "HOLD", "SELL"]
    assert signals[1].outcome_side == "DOWN"
    assert signals[1].unrealized_pnl is None
    assert snapshot.current_round.settlement is not None
    assert snapshot.current_round.settlement.entry_signal == "BUY_UP"
    assert snapshot.current_round.settlement.event_id == "evt-sell-up"
    assert snapshot.current_round.settlement.realized_pnl == pytest.approx(0.10)


def test_dashboard_uses_down_token_edge_for_buy_down(tmp_path: Path) -> None:
    db_path = tmp_path / "mlops.duckdb"
    conn = connect_mlops_db(db_path)
    initialize_mlops_db(conn)
    _record_event(
        conn,
        event_id="evt-down-buy",
        ts=1_000_000,
        prob=0.10,
        market=0.40,
        round_id="round-down",
        outcome_side="DOWN",
    )
    conn.close()

    snapshot = read_dashboard_snapshot(
        db_path,
        model_version="xgboost-v3",
        edge_threshold=0.30,
        outcome_side="DOWN",
        now_ms=1_100_000,
    )

    assert snapshot.current_round is not None
    assert snapshot.current_round.latest_signal.signal_type == "BUY_DOWN"
    assert snapshot.current_round.latest_signal.edge == pytest.approx(0.50)


def _record_event(
    conn,
    *,
    event_id: str,
    ts: int,
    prob: float,
    market: float,
    round_id: str,
    outcome_side: str = "UP",
    features: dict[str, float | None] | None = None,
) -> None:
    feature_values = {"market_implied_prob": market}
    if features:
        feature_values.update(features)
    record_prediction_event(
        conn,
        PredictionEvent(
            event_id=event_id,
            ts=ts,
            model_version="xgboost-v3",
            feature_version="bigan-mvp-v1.0.0",
            prob_up_15m=prob,
            confidence_bucket="high_up",
            top_features_json="[]",
            feature_hash=f"hash-{event_id}",
            feature_snapshot_json=json.dumps(
                {
                    "source_symbol": f"token-{round_id}",
                    "source_market": f"market-{round_id}",
                    "canonical_symbol": f"BTC-15M:{round_id}:{outcome_side}",
                    "market_implied_prob": market,
                    "features": feature_values,
                },
                sort_keys=True,
            ),
            serving_latency_ms=1.0,
            created_at=ts + 100,
        ),
    )


def _record_outcome(
    conn,
    *,
    event_id: str,
    prob: float,
    label: bool,
    realized_return: float,
) -> None:
    record_prediction_outcome(
        conn,
        PredictionOutcome(
            event_id=event_id,
            target_ts=2_000_000,
            realized_label=label,
            realized_return=realized_return,
            brier_component=compute_brier_component(prob, label),
            outcome_ts=2_000_000,
        ),
    )
