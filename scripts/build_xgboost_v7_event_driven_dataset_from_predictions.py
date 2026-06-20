#!/usr/bin/env python3
"""Build a v7 event-driven convergence corpus from prediction warehouse rows.

This is a lightweight rebuild path for already-scored event-driven v7 runs. The
raw low-latency JSONL queue can be very large; the prediction warehouse already
contains bucketed feature payloads plus side quote proxies, so this script
reconstructs the training rows and reuses the existing v7 label attachment
logic.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from bigan.labels.v6 import VolatilityLabelConfig
from bigan.modeling.dataset import _feature_columns

from build_xgboost_v7_event_driven_dataset import (
    DATASET_VERSION,
    LABEL_VERSION,
    _attach_labels,
    _infer_outcomes_from_quotes,
    _split_rows_by_round,
    _split_stats,
    _write_split,
)

SIDES = ("UP", "DOWN")


@dataclass(frozen=True, slots=True)
class PredictionWarehouseBuildStats:
    prediction_glob: str
    bucket_seconds: float
    prediction_rows_read: int
    feature_rows_generated: int
    rows_written: int
    rows_skipped_quality: int
    rows_skipped_unresolved_outcome: int
    rows_skipped_missing_quote: int
    round_count: int
    outcome_counts: dict[str, int]
    unresolved_round_count: int
    splits: dict[str, dict[str, Any]]


def build_event_driven_dataset_from_predictions(
    prediction_glob: Path | str,
    output_dir: Path | str,
    *,
    bucket_seconds: float = 5.0,
    min_completeness_score: float = 0.8,
    allow_unresolved_outcomes: bool = False,
    buy_slippage: float = 0.02,
    sell_slippage: float = 0.02,
    min_exit_gain: float = 0.02,
    min_exit_seconds_before_expiry: float = 300.0,
    min_entry_price: float = 0.0,
) -> PredictionWarehouseBuildStats:
    if bucket_seconds <= 0.0:
        raise ValueError("bucket_seconds must be positive")

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    prediction_glob_str = str(prediction_glob)
    bucket_ms = int(bucket_seconds * 1000)
    if bucket_ms <= 0:
        raise ValueError("bucket_seconds is too small")

    prediction_rows_read = _count_prediction_rows(prediction_glob_str)
    warehouse_rows = _read_bucketed_prediction_rows(prediction_glob_str, bucket_ms=bucket_ms)
    feature_rows, quotes_by_round_side = _feature_rows_and_quotes(warehouse_rows)
    outcomes = _infer_outcomes_from_quotes(quotes_by_round_side)
    labeled_rows, label_stats = _attach_labels(
        feature_rows,
        quotes_by_round_side=quotes_by_round_side,
        outcomes=outcomes,
        min_completeness_score=min_completeness_score,
        require_outcome=not allow_unresolved_outcomes,
        volatility_label_config=VolatilityLabelConfig(
            min_exit_gain=min_exit_gain,
            buy_slippage=buy_slippage,
            sell_slippage=sell_slippage,
            max_entry_wait_ms=max(1, bucket_ms),
            min_exit_seconds_before_expiry=min_exit_seconds_before_expiry,
            min_entry_price=min_entry_price,
        ),
    )
    split_rows = _split_rows_by_round(labeled_rows)
    output_schema = pa.Table.from_pylist(labeled_rows).schema if labeled_rows else pa.schema([])
    for split, rows in split_rows.items():
        _write_split(target / f"{split}.parquet", rows, schema=output_schema)

    stats = PredictionWarehouseBuildStats(
        prediction_glob=prediction_glob_str,
        bucket_seconds=float(bucket_seconds),
        prediction_rows_read=int(prediction_rows_read),
        feature_rows_generated=len(feature_rows),
        rows_written=len(labeled_rows),
        rows_skipped_quality=int(label_stats["rows_skipped_quality"]),
        rows_skipped_unresolved_outcome=int(label_stats["rows_skipped_unresolved_outcome"]),
        rows_skipped_missing_quote=int(label_stats["rows_skipped_missing_quote"]),
        round_count=len({str(row["round_slug"]) for row in labeled_rows}),
        outcome_counts=dict(Counter(outcomes.values())),
        unresolved_round_count=len(set(quotes_by_round_side) - set(outcomes)),
        splits={split: _split_stats(rows) for split, rows in split_rows.items()},
    )
    manifest = {
        "dataset_version": DATASET_VERSION,
        "label_version": LABEL_VERSION,
        "feature_columns": list(_feature_columns()),
        "feature_versions": sorted(
            {
                str(row.get("feature_version"))
                for row in labeled_rows
                if row.get("feature_version") is not None
            }
        ),
        "label_versions": [LABEL_VERSION],
        "bucket_seconds": float(bucket_seconds),
        "row_grain": "event_driven_bucket",
        "source": "prediction_warehouse",
        "settlement_outcome_source": "warehouse_final_quote_inference",
        "convergence_label_formula": (
            "best executable future bid by side minus current side entry worst price"
        ),
        "start_target_price_note": (
            "Rows are reconstructed from event-driven v7 prediction warehouse "
            "feature_values_json; quote paths use warehouse market_implied_prob "
            "with spread-derived bid/ask proxies."
        ),
        "build": asdict(stats),
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return stats


def _count_prediction_rows(prediction_glob: str) -> int:
    con = duckdb.connect()
    try:
        return int(
            con.sql(
                f"""
                SELECT count(*)
                FROM read_parquet({_duckdb_string(prediction_glob)}, hive_partitioning=true, union_by_name=true)
                WHERE canonical_symbol LIKE 'BTC-15M:btc-updown-15m-%'
                """
            ).fetchone()[0]
        )
    finally:
        con.close()


def _read_bucketed_prediction_rows(
    prediction_glob: str,
    *,
    bucket_ms: int,
) -> list[dict[str, Any]]:
    con = duckdb.connect()
    try:
        result = con.sql(
            f"""
            WITH p AS (
              SELECT
                canonical_symbol,
                split_part(canonical_symbol, ':', 2) AS round_slug,
                split_part(canonical_symbol, ':', 3) AS token_side,
                CAST(floor(ts / {bucket_ms}) * {bucket_ms} AS BIGINT) AS feature_ts,
                ts,
                message_ts,
                prediction_ts,
                ingest_ts,
                source,
                source_symbol,
                source_market,
                symbol,
                feature_version,
                market_implied_prob,
                feature_values_json
              FROM read_parquet({_duckdb_string(prediction_glob)}, hive_partitioning=true, union_by_name=true)
              WHERE canonical_symbol LIKE 'BTC-15M:btc-updown-15m-%'
                AND feature_values_json IS NOT NULL
                AND market_implied_prob IS NOT NULL
            ), ranked AS (
              SELECT *,
                     row_number() OVER (
                       PARTITION BY round_slug, token_side, feature_ts
                       ORDER BY ts DESC
                     ) AS rn
              FROM p
              WHERE round_slug <> '' AND token_side IN ('UP', 'DOWN')
            )
            SELECT * EXCLUDE (rn)
            FROM ranked
            WHERE rn = 1
            ORDER BY feature_ts, canonical_symbol
            """
        )
        columns = [desc[0] for desc in result.description]
        return [dict(zip(columns, row, strict=True)) for row in result.fetchall()]
    finally:
        con.close()


def _feature_rows_and_quotes(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, list[dict[str, float | int]]]]]:
    feature_rows: list[dict[str, Any]] = []
    quotes_by_round_side: dict[str, dict[str, list[dict[str, float | int]]]] = {}
    for row in rows:
        round_slug = str(row["round_slug"])
        token_side = str(row["token_side"]).upper()
        feature_ts = _int(row.get("feature_ts"))
        ts = _int(row.get("ts")) or feature_ts
        if token_side not in SIDES or feature_ts is None or ts is None:
            continue
        features = _json_object(row.get("feature_values_json"))
        market = _float(row.get("market_implied_prob"))
        if market is None:
            continue
        bid, ask = _quote_from_features(market, features)
        quotes_by_round_side.setdefault(round_slug, {}).setdefault(token_side, []).append(
            {"ts": feature_ts, "bid": bid, "ask": ask}
        )
        feature_rows.append(
            {
                "ts": ts,
                "message_ts": _int(row.get("message_ts")) or ts,
                "feature_ts": feature_ts,
                "ingest_ts": _int(row.get("ingest_ts") or row.get("prediction_ts")) or ts,
                "source": str(row.get("source") or "polymarket"),
                "source_symbol": str(row.get("source_symbol") or ""),
                "source_market": str(row.get("source_market") or ""),
                "canonical_symbol": str(row.get("canonical_symbol") or ""),
                "symbol": str(row.get("symbol") or row.get("canonical_symbol") or ""),
                "feature_version": str(row.get("feature_version") or "bigan-mvp-v1.0.0"),
                "quote_age_ms": 0,
                "depth_age_ms": 0,
                "trade_age_ms": 0,
                "completeness_score": 1.0,
                "data_gap_flag": False,
                "quality_filter_pass": True,
                **{column: features.get(column) for column in _feature_columns()},
            }
        )
    for by_side in quotes_by_round_side.values():
        for side in SIDES:
            by_side.get(side, []).sort(key=lambda quote: int(quote["ts"]))
    return feature_rows, quotes_by_round_side


def _quote_from_features(market: float, features: Mapping[str, Any]) -> tuple[float, float]:
    spread = _float(features.get("spread"))
    if spread is None:
        spread = _float(features.get("tick_spread"))
    if spread is None or spread < 0.0:
        spread = 0.02
    half_spread = min(0.25, spread / 2.0)
    bid = max(0.001, min(0.999, market - half_spread))
    ask = max(0.001, min(0.999, market + half_spread))
    if ask < bid:
        ask = bid
    return bid, ask


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if value is None:
        return {}
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _duckdb_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-glob", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bucket-seconds", type=float, default=5.0)
    parser.add_argument("--min-completeness-score", type=float, default=0.8)
    parser.add_argument("--allow-unresolved-outcomes", action="store_true")
    parser.add_argument("--buy-slippage", type=float, default=0.02)
    parser.add_argument("--sell-slippage", type=float, default=0.02)
    parser.add_argument("--min-exit-gain", type=float, default=0.02)
    parser.add_argument("--min-exit-seconds-before-expiry", type=float, default=300.0)
    parser.add_argument("--min-entry-price", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    stats = build_event_driven_dataset_from_predictions(
        args.prediction_glob,
        args.output_dir,
        bucket_seconds=args.bucket_seconds,
        min_completeness_score=args.min_completeness_score,
        allow_unresolved_outcomes=args.allow_unresolved_outcomes,
        buy_slippage=args.buy_slippage,
        sell_slippage=args.sell_slippage,
        min_exit_gain=args.min_exit_gain,
        min_exit_seconds_before_expiry=args.min_exit_seconds_before_expiry,
        min_entry_price=args.min_entry_price,
    )
    print(json.dumps(asdict(stats), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
