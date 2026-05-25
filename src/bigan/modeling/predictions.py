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
from bigan.monitoring import (
    prediction_event_from_prediction_row,
    record_prediction_rows_as_events,
)

from .calibration import (
    FamilyAwareProbabilityCalibrator,
    ProbabilityCalibrator,
    load_probability_calibrator,
    transform_probability,
)
from .xgboost_v1 import XGBoostV1Model, load_xgboost_v1_model


@dataclass(frozen=True, slots=True)
class PredictionBatchReport:
    """Summary of a predictions table write."""

    rows_generated: int
    rows_written: int
    model_version: str
    calibration_method: str | None
    monitoring_events_written: int = 0

    def to_dict(self) -> dict[str, int | str | None]:
        return asdict(self)


def generate_prediction_rows(
    *,
    feature_rows: list[dict[str, Any]],
    model: XGBoostV1Model,
    calibrator: ProbabilityCalibrator | FamilyAwareProbabilityCalibrator | None = None,
    ingest_ts: int | None = None,
) -> list[dict[str, Any]]:
    """Generate frontend/API-ready prediction rows from feature rows."""

    run_ingest_ts = int(time.time() * 1000) if ingest_ts is None else int(ingest_ts)
    rows: list[dict[str, Any]] = []
    for feature in feature_rows:
        _validate_training_schema(feature, model.feature_columns)
        prediction_ts = int(feature["feature_ts"])
        raw_probability = model.predict_proba(feature)
        probability = transform_probability(calibrator, raw_probability, feature=feature)
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
                "market_implied_prob": _optional_float(feature.get("market_implied_prob")),
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
    monitoring_db_path: Path | str | None = None,
    since_ms: int | None = None,
    until_ms: int | None = None,
    skip_existing_monitoring_events: bool = False,
    skip_existing_predictions: bool = False,
) -> PredictionBatchReport:
    """Read feature rows from the warehouse and append ``predictions`` rows."""

    model = load_xgboost_v1_model(model_path)
    calibrator = (
        None if calibration_path is None else load_probability_calibrator(calibration_path)
    )
    feature_rows = _read_feature_rows(
        warehouse_dir,
        since_ms=since_ms,
        until_ms=until_ms,
    )
    rows = generate_prediction_rows(
        feature_rows=feature_rows,
        model=model,
        calibrator=calibrator,
        ingest_ts=ingest_ts,
    )
    rows_generated = len(rows)
    if skip_existing_predictions and rows:
        rows = _filter_new_prediction_rows(warehouse_dir, rows)
    with WarehouseWriter(
        warehouse_dir,
        max_rows_per_partition=max_rows_per_partition,
    ) as writer:
        writer.append_rows("predictions", rows)
        writer.flush("predictions")
        rows_written = writer.stats.rows_written.get("predictions", 0)
    monitoring_events_written = 0
    if monitoring_db_path is not None and rows:
        from bigan.mlops.registry import connect_mlops_db, initialize_mlops_db

        conn = connect_mlops_db(monitoring_db_path)
        try:
            initialize_mlops_db(conn)
            monitoring_rows = (
                _filter_new_monitoring_event_rows(conn, rows)
                if skip_existing_monitoring_events
                else rows
            )
            monitoring_events_written = record_prediction_rows_as_events(
                conn,
                monitoring_rows,
            )
        finally:
            conn.close()
    return PredictionBatchReport(
        rows_generated=rows_generated,
        rows_written=rows_written,
        model_version=model.model_version,
        calibration_method=None if calibrator is None else calibrator.method,
        monitoring_events_written=monitoring_events_written,
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


def _read_feature_rows(
    warehouse_dir: Path | str,
    *,
    since_ms: int | None = None,
    until_ms: int | None = None,
) -> list[dict[str, Any]]:
    clauses = ["quality_filter_pass", "not data_gap_flag"]
    params: list[int] = []
    if since_ms is not None:
        clauses.append("feature_ts >= ?")
        params.append(int(since_ms))
    if until_ms is not None:
        clauses.append("feature_ts < ?")
        params.append(int(until_ms))
    where_sql = " AND ".join(clauses)
    with open_warehouse(warehouse_dir) as conn:
        try:
            return conn.execute(
                f"""
                SELECT * EXCLUDE (rn)
                FROM (
                    SELECT *,
                           row_number() OVER (
                               PARTITION BY feature_ts, source, source_symbol
                               ORDER BY ingest_ts DESC, message_ts DESC, ts DESC
                           ) AS rn
                    FROM features_15m_v1
                    WHERE {where_sql}
                )
                WHERE rn = 1
                ORDER BY feature_ts, source, source_symbol
                """,
                params,
            ).to_arrow_table().to_pylist()
        except (duckdb.CatalogException, duckdb.IOException):
            return []


def _filter_new_monitoring_event_rows(
    conn: duckdb.DuckDBPyConnection,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not rows:
        return []
    event_ids = [prediction_event_from_prediction_row(row).event_id for row in rows]
    placeholders = ", ".join("?" for _ in event_ids)
    existing = {
        str(row[0])
        for row in conn.execute(
            f"""
            SELECT event_id
            FROM prediction_events
            WHERE event_id IN ({placeholders})
            """,
            event_ids,
        ).fetchall()
    }
    return [
        row
        for row, event_id in zip(rows, event_ids, strict=True)
        if event_id not in existing
    ]


def _filter_new_prediction_rows(
    warehouse_dir: Path | str,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    min_ts = min(int(row["prediction_ts"]) for row in rows)
    max_ts = max(int(row["prediction_ts"]) for row in rows)
    with open_warehouse(warehouse_dir) as conn:
        try:
            existing = {
                (int(row[0]), str(row[1]), str(row[2]), str(row[3]))
                for row in conn.execute(
                    """
                    SELECT prediction_ts, source, source_symbol, model_version
                    FROM predictions
                    WHERE prediction_ts >= ?
                      AND prediction_ts <= ?
                    """,
                    [min_ts, max_ts],
                ).fetchall()
            }
        except (duckdb.CatalogException, duckdb.IOException):
            existing = set()
    return [
        row
        for row in rows
        if (
            int(row["prediction_ts"]),
            str(row["source"]),
            str(row["source_symbol"]),
            str(row["model_version"]),
        )
        not in existing
    ]


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


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
