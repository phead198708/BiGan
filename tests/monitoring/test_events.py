"""Prediction monitoring table tests for issue #41."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import duckdb
import pytest

from bigan.mlops import connect_mlops_db, initialize_mlops_db
from bigan.monitoring import (
    PredictionEvent,
    PredictionOutcome,
    compute_brier_component,
    initialize_monitoring_tables,
    prediction_event_from_prediction_row,
    prediction_outcome_from_label_row,
    record_label_rows_as_outcomes,
    record_prediction_event,
    record_prediction_outcome,
    record_prediction_rows_as_events,
    summarize_model_monitoring_daily,
)


def _ts(day: str, hour: int = 0) -> int:
    return int(datetime.fromisoformat(f"{day}T{hour:02d}:00:00").replace(tzinfo=UTC).timestamp() * 1000)


def _event(event_id: str, prob: float, *, day: str = "2026-05-20") -> PredictionEvent:
    return PredictionEvent(
        event_id=event_id,
        ts=_ts(day),
        model_version="xgboost-v1",
        feature_version="bigan-mvp-v1.0.0",
        prob_up_15m=prob,
        confidence_bucket="medium_up",
        top_features_json=json.dumps([{"feature": "ret_15m", "contribution": 0.1}]),
        feature_hash=f"hash-{event_id}",
        feature_snapshot_json=json.dumps({"ret_15m": 0.02}),
        serving_latency_ms=12.5,
    )


def test_monitoring_tables_are_created_by_mlops_initializer() -> None:
    conn = connect_mlops_db()
    initialize_mlops_db(conn)

    for table in ("prediction_events", "prediction_outcomes", "model_monitoring_daily"):
        rows = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
        assert rows, f"{table} was not created"

    with pytest.raises(duckdb.ConstraintException):
        conn.execute(
            """
            INSERT INTO prediction_events (
                event_id, ts, model_version, feature_version, prob_up_15m,
                confidence_bucket, top_features_json, feature_hash,
                feature_snapshot_json, serving_latency_ms, created_at
            ) VALUES ('bad', 1, 'm', 'f', 1.5, 'bad', '[]', 'h', '{}', 1, 1)
            """
        )


def test_prediction_events_and_outcomes_are_traceable_by_event_id() -> None:
    conn = connect_mlops_db()
    record_prediction_event(conn, _event("evt-1", 0.80))
    outcome = PredictionOutcome(
        event_id="evt-1",
        target_ts=_ts("2026-05-20", hour=1),
        realized_label=True,
        realized_return=0.03,
        brier_component=compute_brier_component(0.80, True),
        outcome_ts=_ts("2026-05-20", hour=1),
    )
    record_prediction_outcome(conn, outcome)

    row = conn.execute(
        """
        SELECT e.model_version, e.prob_up_15m, o.realized_label, o.brier_component
        FROM prediction_events e
        JOIN prediction_outcomes o USING (event_id)
        WHERE event_id = 'evt-1'
        """
    ).fetchone()
    assert row == ("xgboost-v1", 0.80, True, pytest.approx(0.04))


def test_prediction_rows_are_recorded_as_monitoring_events() -> None:
    conn = connect_mlops_db()
    row = {
        "prediction_ts": _ts("2026-05-20", hour=1),
        "source": "polymarket",
        "source_symbol": "tok-up",
        "source_market": "mkt-1",
        "canonical_symbol": "BTC-UP-15M",
        "symbol": "BTC-UP-15M",
        "feature_version": "bigan-mvp-v1.0.0",
        "model_version": "xgboost-v3",
        "prob_up_15m": 0.72,
        "market_implied_prob": 0.41,
        "confidence_bucket": "high_up",
        "top_features_json": "[]",
        "feature_values_json": json.dumps({"spread": 0.02}, sort_keys=True),
    }

    event = prediction_event_from_prediction_row(row, serving_latency_ms=1.5)
    assert event.event_id.startswith("pred-")
    assert event.prob_up_15m == pytest.approx(0.72)
    assert json.loads(event.feature_snapshot_json)["market_implied_prob"] == pytest.approx(0.41)

    assert record_prediction_rows_as_events(conn, [row]) == 1
    stored = conn.execute(
        """
        SELECT model_version, prob_up_15m, serving_latency_ms, feature_snapshot_json
        FROM prediction_events
        """
    ).fetchone()
    assert stored[0] == "xgboost-v3"
    assert stored[1] == pytest.approx(0.72)
    assert stored[2] == pytest.approx(0.0)
    assert json.loads(stored[3])["features"] == {"spread": 0.02}


def test_label_rows_are_recorded_as_prediction_outcomes() -> None:
    conn = connect_mlops_db()
    prediction_row = {
        "prediction_ts": _ts("2026-05-20", hour=1),
        "source": "polymarket",
        "source_symbol": "tok-up",
        "source_market": "mkt-1",
        "canonical_symbol": "BTC-UP-15M",
        "symbol": "BTC-UP-15M",
        "feature_version": "bigan-mvp-v1.0.0",
        "model_version": "xgboost-v3",
        "prob_up_15m": 0.72,
        "market_implied_prob": 0.41,
        "confidence_bucket": "high_up",
        "top_features_json": "[]",
        "feature_values_json": "{}",
    }
    record_prediction_rows_as_events(conn, [prediction_row])
    label_row = {
        "feature_ts": prediction_row["prediction_ts"],
        "target_ts": prediction_row["prediction_ts"] + 900_000,
        "ingest_ts": prediction_row["prediction_ts"] + 901_000,
        "source": "polymarket",
        "source_symbol": "tok-up",
        "label_profit_up_15m": True,
        "realized_return": 0.58,
    }

    outcome = prediction_outcome_from_label_row(
        conn,
        label_row,
        model_version="xgboost-v3",
    )
    assert outcome is not None
    assert outcome.brier_component == pytest.approx((0.72 - 1.0) ** 2)

    assert record_label_rows_as_outcomes(
        conn,
        [label_row],
        model_version="xgboost-v3",
    ) == 1
    stored = conn.execute(
        """
        SELECT realized_label, realized_return, brier_component
        FROM prediction_outcomes
        """
    ).fetchone()
    assert stored == (True, pytest.approx(0.58), pytest.approx((0.72 - 1.0) ** 2))


def test_down_label_rows_use_down_token_probability_for_brier() -> None:
    conn = connect_mlops_db()
    prediction_row = {
        "prediction_ts": _ts("2026-05-20", hour=1),
        "source": "polymarket",
        "source_symbol": "tok-down",
        "source_market": "mkt-1",
        "canonical_symbol": "BTC-15M:btc-updown-15m-test:DOWN",
        "symbol": "BTC-15M:btc-updown-15m-test:DOWN",
        "feature_version": "bigan-mvp-v1.0.0",
        "model_version": "xgboost-v3",
        "prob_up_15m": 0.20,
        "market_implied_prob": 0.40,
        "confidence_bucket": "medium_up",
        "top_features_json": "[]",
        "feature_values_json": "{}",
    }
    record_prediction_rows_as_events(conn, [prediction_row])
    label_row = {
        "feature_ts": prediction_row["prediction_ts"],
        "target_ts": prediction_row["prediction_ts"] + 900_000,
        "ingest_ts": prediction_row["prediction_ts"] + 901_000,
        "source": "polymarket",
        "source_symbol": "tok-down",
        "canonical_symbol": "BTC-15M:btc-updown-15m-test:DOWN",
        "label_kind": "down_token_profitability",
        "label_profit_down_15m": True,
        "label_down_15m": True,
        "realized_return": 0.60,
    }

    outcome = prediction_outcome_from_label_row(
        conn,
        label_row,
        model_version="xgboost-v3",
    )

    assert outcome is not None
    assert outcome.realized_label is True
    assert outcome.brier_component == pytest.approx((0.80 - 1.0) ** 2)


def test_daily_monitoring_summary_computes_hit_rate_brier_and_ece() -> None:
    conn = connect_mlops_db()
    initialize_monitoring_tables(conn)
    events = [
        _event("evt-1", 0.80),
        _event("evt-2", 0.30),
        _event("evt-3", 0.60),
    ]
    labels = [True, False, False]
    for event, label in zip(events, labels, strict=True):
        record_prediction_event(conn, event)
        record_prediction_outcome(
            conn,
            PredictionOutcome(
                event_id=event.event_id,
                target_ts=event.ts + 900_000,
                realized_label=label,
                realized_return=0.01 if label else -0.01,
                brier_component=compute_brier_component(event.prob_up_15m, label),
                outcome_ts=event.ts + 900_000,
            ),
        )

    row = summarize_model_monitoring_daily(
        conn,
        date="2026-05-20",
        model_version="xgboost-v1",
        drift_metrics_json=json.dumps({"psi": {"ret_15m": 0.02}}),
        bins=2,
    )

    assert row.prediction_count == 3
    assert row.avg_prob == pytest.approx((0.80 + 0.30 + 0.60) / 3)
    assert row.hit_rate == pytest.approx(2 / 3)
    assert row.brier_score == pytest.approx((0.04 + 0.09 + 0.36) / 3)
    assert row.ece is not None
    stored = conn.execute(
        'SELECT prediction_count, brier_score FROM model_monitoring_daily WHERE "date" = CAST(? AS DATE)',
        ["2026-05-20"],
    ).fetchone()
    assert stored == (3, pytest.approx((0.04 + 0.09 + 0.36) / 3))


def test_prediction_event_validation_rejects_bad_json() -> None:
    conn = connect_mlops_db()
    bad = _event("evt-bad", 0.5)
    bad = PredictionEvent(
        **{**bad.to_row(), "created_at": None, "feature_snapshot_json": "{not-json"}
    )
    with pytest.raises(ValueError, match="feature_snapshot_json"):
        record_prediction_event(conn, bad)
