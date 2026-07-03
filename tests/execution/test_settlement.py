"""Settlement reconciliation tests for issue #75."""

from __future__ import annotations

import duckdb
import pytest

from bigan.execution import (
    PositionManager,
    RiskConfig,
    RiskManager,
    SettlementPoller,
    SettlementPollerConfig,
    read_execution_settlements,
    realized_pnl_for_position,
)
from bigan.monitoring import (
    ChampionDriftThresholds,
    PredictionEvent,
    open_data_quality_incidents,
    record_prediction_event,
)


class FakeResultProvider:
    def __init__(self, *results: str | None) -> None:
        self.results = list(results)
        self.calls = 0

    def get_settlement_result(self, event_id: str) -> str | None:
        self.calls += 1
        if not self.results:
            return None
        return self.results.pop(0)


def _event(event_id: str, *, prob: float = 0.80) -> PredictionEvent:
    return PredictionEvent(
        event_id=event_id,
        ts=1_000,
        model_version="xgboost-v4",
        feature_version="bigan-mvp-v1.0.0",
        prob_up_15m=prob,
        confidence_bucket="high",
        top_features_json="[]",
        feature_hash=f"hash-{event_id}",
        feature_snapshot_json="{}",
        serving_latency_ms=1.0,
    )


def test_settlement_records_up_win_and_updates_risk_and_monitoring(tmp_path) -> None:
    conn = duckdb.connect()
    positions = PositionManager(tmp_path / "positions.duckdb")
    positions.open_position(
        "evt-up-win",
        "BTC-15M:round:UP",
        "UP",
        0.52,
        2.0,
        "order-up",
        fill_price=0.50,
    )
    risk = RiskManager(
        positions,
        config=RiskConfig(max_concurrent_positions=10),
        incident_conn=conn,
    )
    record_prediction_event(conn, _event("evt-up-win", prob=0.90))
    poller = SettlementPoller(
        position_manager=positions,
        result_provider=FakeResultProvider("UP"),
        model_version="xgboost-v4",
        conn=conn,
        risk_manager=risk,
        config=SettlementPollerConfig(poll_interval_seconds=0.0, max_attempts=1),
    )

    result = poller.reconcile_event("evt-up-win", settlement_ts=2_000)

    assert result.record.reason == "settled"
    assert result.record.realized_pnl == pytest.approx(1.0)
    assert result.monitoring_outcome_written is True
    assert risk.get_daily_stats().realized_pnl_usdc == pytest.approx(1.0)
    rows = read_execution_settlements(conn, event_id="evt-up-win")
    assert rows[0]["settlement_result"] == "UP"
    outcome = conn.execute(
        "SELECT realized_label, realized_return FROM prediction_outcomes WHERE event_id = ?",
        ["evt-up-win"],
    ).fetchone()
    assert outcome == (True, pytest.approx(1.0))


def test_settlement_records_up_loss_from_fill_price(tmp_path) -> None:
    conn = duckdb.connect()
    positions = PositionManager(tmp_path / "positions.duckdb")
    positions.open_position(
        "evt-up-loss",
        "BTC-15M:round:UP",
        "UP",
        0.53,
        3.0,
        "order-up-loss",
        fill_price=0.40,
    )
    poller = SettlementPoller(
        position_manager=positions,
        result_provider=FakeResultProvider("DOWN"),
        model_version="xgboost-v4",
        conn=conn,
        config=SettlementPollerConfig(poll_interval_seconds=0.0, max_attempts=1),
    )

    result = poller.reconcile_event("evt-up-loss", settlement_ts=2_000)

    assert result.record.realized_pnl == pytest.approx(-1.2)
    assert positions.get_position("evt-up-loss").realized_pnl == pytest.approx(-1.2)


def test_early_sell_uses_exit_price_without_polling_settlement(tmp_path) -> None:
    conn = duckdb.connect()
    positions = PositionManager(tmp_path / "positions.duckdb")
    positions.open_position(
        "evt-sell",
        "BTC-15M:round:UP",
        "UP",
        0.51,
        2.0,
        "order-sell",
        fill_price=0.50,
    )
    positions.close_position("evt-sell", 0.62, exit_time=1_500)
    provider = FakeResultProvider("DOWN")
    poller = SettlementPoller(
        position_manager=positions,
        result_provider=provider,
        model_version="xgboost-v4",
        conn=conn,
        config=SettlementPollerConfig(poll_interval_seconds=0.0, max_attempts=1),
    )

    result = poller.reconcile_event("evt-sell", settlement_ts=2_000)

    assert provider.calls == 0
    assert result.record.reason == "early_sell"
    assert result.record.realized_pnl == pytest.approx(0.24)
    assert realized_pnl_for_position(positions.get_position("evt-sell")) == pytest.approx(0.24)


def test_timeout_records_manual_review_row(tmp_path) -> None:
    conn = duckdb.connect()
    positions = PositionManager(tmp_path / "positions.duckdb")
    positions.open_position("evt-timeout", "BTC-15M:round:DOWN", "DOWN", 0.50, 1.0, "order-timeout")
    provider = FakeResultProvider(None, None, None)
    sleeps: list[float] = []
    poller = SettlementPoller(
        position_manager=positions,
        result_provider=provider,
        model_version="xgboost-v4",
        conn=conn,
        config=SettlementPollerConfig(poll_interval_seconds=0.0, max_attempts=3),
        sleep=sleeps.append,
    )

    result = poller.reconcile_event("evt-timeout", settlement_ts=2_000)

    assert provider.calls == 3
    assert sleeps == [0.0, 0.0]
    assert result.record.reason == "timeout"
    assert result.record.settlement_result == "TIMEOUT"
    assert result.record.realized_pnl == 0.0


def test_no_entry_round_records_zero_pnl_and_complete_denominator(tmp_path) -> None:
    conn = duckdb.connect()
    positions = PositionManager(tmp_path / "positions.duckdb")
    record_prediction_event(conn, _event("evt-hold", prob=0.20))
    poller = SettlementPoller(
        position_manager=positions,
        result_provider=FakeResultProvider("DOWN"),
        model_version="xgboost-v4",
        conn=conn,
        config=SettlementPollerConfig(poll_interval_seconds=0.0, max_attempts=1),
    )

    result = poller.reconcile_event("evt-hold", settlement_ts=2_000)

    assert result.record.reason == "no_entry"
    assert result.record.side is None
    assert result.record.realized_pnl == 0.0
    assert result.monitoring_outcome_written is True
    rows = read_execution_settlements(conn)
    assert len(rows) == 1
    assert rows[0]["reason"] == "no_entry"


def test_label_hit_rate_alert_is_recorded_after_settlement(tmp_path) -> None:
    conn = duckdb.connect()
    positions = PositionManager(tmp_path / "positions.duckdb")
    thresholds = ChampionDriftThresholds(label_consecutive_samples=2)
    for event_id in ("evt-loss-1", "evt-loss-2"):
        positions.open_position(event_id, "BTC-15M:round:UP", "UP", 0.50, 1.0, f"order-{event_id}")
        record_prediction_event(conn, _event(event_id, prob=0.90))
        poller = SettlementPoller(
            position_manager=positions,
            result_provider=FakeResultProvider("DOWN"),
            model_version="xgboost-v4",
            conn=conn,
            config=SettlementPollerConfig(poll_interval_seconds=0.0, max_attempts=1),
            label_thresholds=thresholds,
        )
        result = poller.reconcile_event(event_id, settlement_ts=2_000)

    assert result.label_hit_rate["alert"] is True
    assert result.incident_ids
    incidents = open_data_quality_incidents(conn, severity="critical")
    assert incidents[0]["incident_type"] == "label_shift"
