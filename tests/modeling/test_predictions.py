"""TDD contract for issue #20 online prediction output contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import xgboost as xgb

from bigan.canonical.writer import WarehouseWriter, warehouse_files


def _feature_row(ts: int, mid_price: float) -> dict:
    return {
        "source": "polymarket",
        "source_symbol": "tok-up",
        "source_market": "0xmkt",
        "canonical_symbol": "BTC-UP-15M",
        "symbol": "BTC-UP-15M",
        "feature_ts": ts,
        "feature_version": "bigan-mvp-v1.0.0",
        "spread": 0.02,
        "market_implied_prob": mid_price + 0.01,
        "mid_price": mid_price,
        "ret_15m": mid_price - 0.50,
    }


def _warehouse_feature_row(ts: int, mid_price: float, *, ingest_ts: int | None = None) -> dict:
    return {
        **_feature_row(ts, mid_price),
        "ts": ts,
        "message_ts": ts,
        "ingest_ts": ts if ingest_ts is None else ingest_ts,
        "completeness_score": 1.0,
        "data_gap_flag": False,
        "quality_filter_pass": True,
        "quote_age_ms": 0,
        "depth_age_ms": 0,
        "trade_age_ms": None,
        "microprice": mid_price,
        "obi_l1": 0.0,
        "obi_l5": 0.0,
        "obi_l10": 0.0,
        "signed_volume_1m": 0.0,
        "trade_imbalance_1m": None,
        "trade_count_1m": 0,
        "trade_volume_1m": None,
        "ret_1m": None,
        "ret_5m": None,
        "rv_1m": None,
        "rv_5m": None,
        "rv_15m": None,
    }


def _model():
    from bigan.modeling import XGBoostV1Model

    feature_columns = ("spread", "mid_price", "ret_15m")
    matrix = np.asarray(
        [
            [0.02, 0.40, -0.10],
            [0.02, 0.45, -0.05],
            [0.02, 0.55, 0.05],
            [0.02, 0.60, 0.10],
        ],
        dtype=float,
    )
    labels = np.asarray([0, 0, 1, 1], dtype=float)
    booster = xgb.train(
        {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "seed": 0,
            "nthread": 1,
            "eta": 1.0,
            "max_depth": 2,
            "min_child_weight": 0,
        },
        xgb.DMatrix(matrix, label=labels, feature_names=list(feature_columns)),
        num_boost_round=3,
        verbose_eval=False,
    )
    booster.set_attr(
        model_version="xgboost-v1",
        feature_columns=json.dumps(list(feature_columns)),
        params=json.dumps({"max_depth": 2, "rounds": 3}),
    )
    return XGBoostV1Model(
        model_version="xgboost-v1",
        feature_columns=feature_columns,
        booster=booster,
        params={"max_depth": 2, "rounds": 3},
    )


def test_predictions_schema_is_registered_with_online_contract_fields() -> None:
    from bigan.canonical.schemas import SCHEMAS, TABLE_NAMES

    assert "predictions" in TABLE_NAMES
    schema = SCHEMAS["predictions"]
    cols = {field.name for field in schema}

    assert {
        "ts",
        "message_ts",
        "prediction_ts",
        "ingest_ts",
        "source",
        "source_symbol",
        "source_market",
        "canonical_symbol",
        "symbol",
        "feature_version",
        "model_version",
        "calibration_method",
        "prob_up_15m",
        "raw_prob_up_15m",
        "market_implied_prob",
        "confidence_bucket",
        "top_features_json",
    } <= cols
    assert schema.field("prediction_ts").type == pa.int64()
    assert schema.field("prob_up_15m").type == pa.float64()
    assert schema.field("market_implied_prob").type == pa.float64()
    assert schema.field("model_version").type == pa.string()
    assert not schema.field("prob_up_15m").nullable


def test_generate_prediction_rows_outputs_frontend_ready_contract(tmp_path: Path) -> None:
    from bigan.modeling import ProbabilityCalibrator, generate_prediction_rows

    calibrator = ProbabilityCalibrator(
        method="isotonic",
        model_version="xgboost-v1",
        params={"blocks": [{"max_probability": 1.0, "value": 0.90}]},
    )
    rows = generate_prediction_rows(
        feature_rows=[_feature_row(1_800_000, 0.60)],
        model=_model(),
        calibrator=calibrator,
        ingest_ts=1_800_123,
    )

    row = rows[0]
    assert row["ts"] == 1_800_000
    assert row["prediction_ts"] == 1_800_000
    assert row["model_version"] == "xgboost-v1"
    assert row["calibration_method"] == "isotonic"
    assert row["prob_up_15m"] == pytest.approx(0.90)
    assert 0.0 <= row["raw_prob_up_15m"] <= 1.0
    assert row["market_implied_prob"] == pytest.approx(0.61)
    assert row["confidence_bucket"] == "high_up"
    assert json.loads(row["top_features_json"])[0]["feature"] == "mid_price"

    warehouse = tmp_path / "warehouse"
    with WarehouseWriter(warehouse, max_rows_per_partition=10) as writer:
        writer.append_rows("predictions", rows)

    files = warehouse_files(warehouse, "predictions")
    assert files
    stored = pq.ParquetFile(files[0]).read().to_pylist()[0]
    assert stored["prob_up_15m"] == pytest.approx(0.90)
    assert stored["market_implied_prob"] == pytest.approx(0.61)


def test_generate_prediction_rows_rejects_training_schema_mismatch() -> None:
    from bigan.modeling import generate_prediction_rows

    with pytest.raises(ValueError, match="training schema"):
        generate_prediction_rows(
            feature_rows=[{"source": "polymarket", "source_symbol": "tok-up", "feature_ts": 1}],
            model=_model(),
        )


def test_run_prediction_batch_can_score_recent_features_and_skip_existing_events(
    tmp_path: Path,
) -> None:
    from bigan.mlops.registry import connect_mlops_db, initialize_mlops_db
    from bigan.modeling import run_prediction_batch

    warehouse = tmp_path / "warehouse"
    model_path = tmp_path / "model.json"
    mlops_db = tmp_path / "mlops.duckdb"
    model = _model()
    model.booster.save_model(model_path)
    with WarehouseWriter(warehouse, max_rows_per_partition=10) as writer:
        writer.append_rows("features_15m_v1", [_warehouse_feature_row(1_000, 0.40)])
        writer.append_rows(
            "features_15m_v1",
            [
                _warehouse_feature_row(2_000, 0.50, ingest_ts=10),
                _warehouse_feature_row(2_000, 0.60, ingest_ts=20),
            ],
        )

    first = run_prediction_batch(
        warehouse,
        model_path,
        since_ms=2_000,
        monitoring_db_path=mlops_db,
        skip_existing_monitoring_events=True,
    )
    second = run_prediction_batch(
        warehouse,
        model_path,
        since_ms=2_000,
        monitoring_db_path=mlops_db,
        skip_existing_monitoring_events=True,
    )

    assert first.rows_generated == 1
    assert first.monitoring_events_written == 1
    assert second.rows_generated == 1
    assert second.monitoring_events_written == 0
    conn = connect_mlops_db(mlops_db)
    try:
        initialize_mlops_db(conn)
        event = conn.execute(
            "select ts, feature_snapshot_json from prediction_events"
        ).fetchone()
    finally:
        conn.close()
    assert event[0] == 2_000
    snapshot = json.loads(event[1])
    assert snapshot["features"]["mid_price"] == pytest.approx(0.60)


def test_run_prediction_batch_can_skip_existing_prediction_rows(tmp_path: Path) -> None:
    from bigan.modeling import run_prediction_batch

    warehouse = tmp_path / "warehouse"
    model_path = tmp_path / "model.json"
    model = _model()
    model.booster.save_model(model_path)
    with WarehouseWriter(warehouse, max_rows_per_partition=10) as writer:
        writer.append_rows("features_15m_v1", [_warehouse_feature_row(2_000, 0.50)])

    first = run_prediction_batch(
        warehouse,
        model_path,
        since_ms=2_000,
        skip_existing_predictions=True,
    )
    second = run_prediction_batch(
        warehouse,
        model_path,
        since_ms=2_000,
        skip_existing_predictions=True,
    )

    assert first.rows_generated == 1
    assert first.rows_written == 1
    assert second.rows_generated == 1
    assert second.rows_written == 0


def test_run_prediction_batch_can_filter_by_canonical_symbol(tmp_path: Path) -> None:
    from bigan.modeling import run_prediction_batch

    warehouse = tmp_path / "warehouse"
    model_path = tmp_path / "model.json"
    model = _model()
    model.booster.save_model(model_path)
    btc = {
        **_warehouse_feature_row(2_000, 0.50),
        "source_symbol": "btc-up-token",
        "canonical_symbol": "BTC-15M:btc-round:UP",
        "symbol": "BTC-15M:btc-round:UP",
    }
    eth = {
        **_warehouse_feature_row(2_000, 0.60),
        "source_symbol": "eth-up-token",
        "canonical_symbol": "ETH-15M:eth-round:UP",
        "symbol": "ETH-15M:eth-round:UP",
    }
    with WarehouseWriter(warehouse, max_rows_per_partition=10) as writer:
        writer.append_rows("features_15m_v1", [btc, eth])

    report = run_prediction_batch(
        warehouse,
        model_path,
        since_ms=2_000,
        canonical_symbol_like="BTC-15M:%",
    )

    assert report.rows_generated == 1
    files = warehouse_files(warehouse, "predictions")
    row = pq.ParquetFile(files[0]).read().to_pylist()[0]
    assert row["source_symbol"] == "btc-up-token"
    assert row["canonical_symbol"] == "BTC-15M:btc-round:UP"
