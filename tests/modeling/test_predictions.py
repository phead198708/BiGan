"""TDD contract for issue #20 online prediction output contract."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

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
        "mid_price": mid_price,
        "ret_15m": mid_price - 0.50,
    }


def _model():
    from bigan.modeling import XGBoostV1Model, XGBoostV1Stump

    return XGBoostV1Model(
        model_version="xgboost-v1",
        feature_columns=("spread", "mid_price", "ret_15m"),
        base_score=0.0,
        stumps=(
            XGBoostV1Stump(
                feature="mid_price",
                threshold=0.50,
                left_value=-1.0,
                right_value=1.0,
                gain=2.0,
            ),
        ),
        feature_means={"spread": 0.02, "mid_price": 0.50, "ret_15m": 0.0},
        params={"rounds": 1, "learning_rate": 1.0, "l2_penalty": 0.0, "max_depth": 1},
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
        "confidence_bucket",
        "top_features_json",
    } <= cols
    assert schema.field("prediction_ts").type == pa.int64()
    assert schema.field("prob_up_15m").type == pa.float64()
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
    assert row["confidence_bucket"] == "high_up"
    assert json.loads(row["top_features_json"])[0]["feature"] == "mid_price"

    warehouse = tmp_path / "warehouse"
    with WarehouseWriter(warehouse, max_rows_per_partition=10) as writer:
        writer.append_rows("predictions", rows)

    files = warehouse_files(warehouse, "predictions")
    assert files
    stored = pq.ParquetFile(files[0]).read().to_pylist()[0]
    assert stored["prob_up_15m"] == pytest.approx(0.90)


def test_generate_prediction_rows_rejects_training_schema_mismatch() -> None:
    from bigan.modeling import generate_prediction_rows

    with pytest.raises(ValueError, match="training schema"):
        generate_prediction_rows(
            feature_rows=[{"source": "polymarket", "source_symbol": "tok-up", "feature_ts": 1}],
            model=_model(),
        )
