"""Online prediction output contract (issue #20)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb

from bigan.canonical.query import open_warehouse
from bigan.canonical.writer import WarehouseWriter

from .calibration import ProbabilityCalibrator, load_probability_calibrator
from .xgboost_v1 import XGBoostV1Model, load_xgboost_v1_model


@dataclass(frozen=True, slots=True)
class PredictionBatchReport:
    """Summary of a predictions table write."""

    rows_generated: int
    rows_written: int
    model_version: str
    calibration_method: str | None

    def to_dict(self) -> dict[str, int | str | None]:
        return asdict(self)


def generate_prediction_rows(
    *,
    feature_rows: list[dict[str, Any]],
    model: XGBoostV1Model,
    calibrator: ProbabilityCalibrator | None = None,
    ingest_ts: int | None = None,
) -> list[dict[str, Any]]:
    """Generate frontend/API-ready prediction rows from feature rows."""

    run_ingest_ts = int(time.time() * 1000) if ingest_ts is None else int(ingest_ts)
    rows: list[dict[str, Any]] = []
    for feature in feature_rows:
        _validate_training_schema(feature, model.feature_columns)
        prediction_ts = int(feature["feature_ts"])
        raw_probability = model.predict_proba(feature)
        probability = calibrator.transform(raw_probability) if calibrator is not None else raw_probability
        top_features = model.top_feature_contributions(feature)
        rows.append(
            {
                "ts": prediction_ts,
                "message_ts": prediction_ts,
                "prediction_ts": prediction_ts,
                "ingest_ts": run_ingest_ts,
                "source": str(feature["source"]),
                "source_symbol": str(feature["source_symbol"]),
                "source_market": _optional_str(feature.get("source_market")),
                "canonical_symbol": _optional_str(feature.get("canonical_symbol")),
                "symbol": str(
                    feature.get("symbol")
                    or feature.get("canonical_symbol")
                    or feature["source_symbol"]
                ),
                "feature_version": str(feature["feature_version"]),
                "model_version": model.model_version,
                "calibration_method": None if calibrator is None else calibrator.method,
                "prob_up_15m": probability,
                "raw_prob_up_15m": raw_probability,
                "confidence_bucket": confidence_bucket(probability),
                "top_features_json": json.dumps(top_features, sort_keys=True),
                "feature_values_json": json.dumps(
                    {
                        column: feature.get(column)
                        for column in model.feature_columns
                    },
                    sort_keys=True,
                ),
            }
        )
    rows.sort(key=lambda row: (row["prediction_ts"], row["source"], row["source_symbol"]))
    return rows


def run_prediction_batch(
    warehouse_dir: Path | str,
    model_path: Path | str,
    *,
    calibration_path: Path | str | None = None,
    max_rows_per_partition: int = 50_000,
    ingest_ts: int | None = None,
) -> PredictionBatchReport:
    """Read feature rows from the warehouse and append ``predictions`` rows."""

    model = load_xgboost_v1_model(model_path)
    calibrator = (
        None if calibration_path is None else load_probability_calibrator(calibration_path)
    )
    feature_rows = _read_feature_rows(warehouse_dir)
    rows = generate_prediction_rows(
        feature_rows=feature_rows,
        model=model,
        calibrator=calibrator,
        ingest_ts=ingest_ts,
    )
    with WarehouseWriter(
        warehouse_dir,
        max_rows_per_partition=max_rows_per_partition,
    ) as writer:
        writer.append_rows("predictions", rows)
        writer.flush("predictions")
        rows_written = writer.stats.rows_written.get("predictions", 0)
    return PredictionBatchReport(
        rows_generated=len(rows),
        rows_written=rows_written,
        model_version=model.model_version,
        calibration_method=None if calibrator is None else calibrator.method,
    )


def confidence_bucket(probability: float) -> str:
    """Map probability to a stable UI/API confidence bucket."""

    if probability >= 0.65:
        return "high_up"
    if probability >= 0.55:
        return "medium_up"
    if probability <= 0.35:
        return "high_down"
    if probability <= 0.45:
        return "medium_down"
    return "neutral"


def _read_feature_rows(warehouse_dir: Path | str) -> list[dict[str, Any]]:
    with open_warehouse(warehouse_dir) as conn:
        try:
            return conn.execute(
                """
                select *
                from features_15m_v1
                where quality_filter_pass
                  and not data_gap_flag
                order by feature_ts, source, source_symbol
                """
            ).to_arrow_table().to_pylist()
        except (duckdb.CatalogException, duckdb.IOException):
            return []


def _validate_training_schema(row: dict[str, Any], feature_columns: tuple[str, ...]) -> None:
    missing = sorted(column for column in feature_columns if column not in row)
    identity_missing = sorted(
        column
        for column in ("source", "source_symbol", "feature_ts", "feature_version")
        if column not in row
    )
    if missing or identity_missing:
        details = ", ".join(missing + identity_missing)
        raise ValueError(f"online row does not match training schema: missing {details}")


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)
