"""Serving readiness evidence tests."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _sample(feature_ts: int, mid_price: float, label: bool) -> dict:
    return {
        "feature_ts": feature_ts,
        "feature_version": "bigan-mvp-v1.0.0",
        "target_ts": feature_ts + 900_000,
        "source": "polymarket",
        "source_symbol": "tok-up",
        "label_profit_up_15m": label,
        "spread": 0.02 + abs(mid_price - 0.50) / 10,
        "mid_price": mid_price,
        "ret_15m": mid_price - 0.50,
    }


def _write_dataset(dataset_dir: Path) -> None:
    dataset_dir.mkdir(parents=True)
    rows = [
        _sample(idx * 60_000, mid_price, label=mid_price >= 0.50)
        for idx, mid_price in enumerate((0.40, 0.45, 0.55, 0.60, 0.42, 0.58))
    ]
    for split in ("train", "val", "test"):
        pq.write_table(pa.Table.from_pylist(rows), dataset_dir / f"{split}.parquet")
    (dataset_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": "bigan-training-15m-profitability-v1.0.0",
                "feature_columns": ["spread", "mid_price", "ret_15m"],
                "feature_versions": ["bigan-mvp-v1.0.0"],
            }
        ),
        encoding="utf-8",
    )


def _v6_sample(feature_ts: int, mid_price: float, settlement: str, idx: int) -> dict:
    up_vol = settlement == "UP" or idx % 3 == 0
    down_vol = settlement == "DOWN" or idx % 4 == 0
    return {
        "source": "polymarket",
        "source_symbol": f"tok-{idx}",
        "source_market": "0xmkt",
        "canonical_symbol": "BTC-15M:btc-updown-15m-1000:UP",
        "symbol": "BTC-15M:btc-updown-15m-1000:UP",
        "feature_ts": feature_ts,
        "feature_version": "bigan-mvp-v1.0.0",
        "label_version": "bigan-labels-v6.0.0",
        "target_ts": feature_ts + 900_000,
        "round_start_ts": feature_ts - 60_000,
        "round_end_ts": feature_ts + 900_000,
        "start_price": 100.0,
        "target_price": 101.0 if settlement == "UP" else 99.0 if settlement == "DOWN" else 100.0,
        "label_up_15m": settlement == "UP",
        "label_settlement_3way": settlement,
        "label_volatility_up": up_vol,
        "label_volatility_down": down_vol,
        "max_exit_gain_up": 0.24 + (0.06 if up_vol else 0.0),
        "max_exit_gain_down": 0.22 + (0.06 if down_vol else 0.0),
        "realized_return": 0.40 if settlement == "UP" else -0.40,
        "completeness_score": 1.0,
        "data_gap_flag": False,
        "quality_filter_pass": True,
        "spread": 0.02 + abs(mid_price - 0.50) / 10,
        "mid_price": mid_price,
        "market_implied_prob": mid_price,
        "underlying_id": 0.0,
        "horizon_minutes": 15.0,
        "liquidity_bucket": 1.0,
        "ret_15m": mid_price - 0.50,
        "minute_of_day": ((feature_ts // 60_000) % 1440) / 1439,
        "day_of_week": 2,
        "ret_30m": 2 * (mid_price - 0.50),
        "rv_30m": abs(mid_price - 0.50) + (0.05 if up_vol or down_vol else 0.0),
        "aggressor_buy_ratio_1m": 0.75 if mid_price >= 0.50 else 0.25,
        "avg_trade_size_1m": 10.0 + mid_price,
        "tick_spread": 0.02 + abs(mid_price - 0.50) / 10,
        "tick_obi_l1": mid_price - 0.50,
        "tick_obi_l3": (mid_price - 0.50) / 2,
        "tick_mid_price": mid_price,
        "tick_price_velocity": mid_price - 0.50,
        "tick_trade_arrival_rate": 3.0 + abs(mid_price - 0.50),
        "v5_prob_up_15m": min(0.95, max(0.05, mid_price)),
    }


def _write_v6_dataset(dataset_dir: Path) -> None:
    from bigan.modeling import (
        XGBOOST_V4_REQUIRED_ADDED_FEATURES,
        XGBOOST_V4_REQUIRED_MARKET_FEATURES,
        XGBOOST_V4_REQUIRED_TICK_FEATURES,
    )

    dataset_dir.mkdir(parents=True)
    feature_columns = [
        "spread",
        "mid_price",
        "market_implied_prob",
        "ret_15m",
        *XGBOOST_V4_REQUIRED_MARKET_FEATURES,
        *XGBOOST_V4_REQUIRED_ADDED_FEATURES,
        *XGBOOST_V4_REQUIRED_TICK_FEATURES,
    ]
    settlements = ["DOWN", "NEUTRAL", "UP", "DOWN", "NEUTRAL", "UP"]
    rows = [
        _v6_sample(idx * 60_000, mid_price, settlements[idx % len(settlements)], idx)
        for idx, mid_price in enumerate([0.34, 0.45, 0.64, 0.38, 0.50, 0.68])
    ]
    for split in ("train", "val", "test"):
        pq.write_table(pa.Table.from_pylist(rows), dataset_dir / f"{split}.parquet")
    (dataset_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": "bigan-training-v6-serving-test",
                "feature_columns": feature_columns,
                "v5_feature_columns": feature_columns,
                "feature_versions": ["bigan-mvp-v1.0.0"],
                "label_versions": ["bigan-labels-v6.0.0"],
                "expected_sample_count_per_family": {"BTC-15M": 6},
                "v6_label_diagnostics": {"phase4_capture_rows": 6},
                "rows_written": 6,
            }
        ),
        encoding="utf-8",
    )


def test_xgboost_serving_readiness_writes_latency_schema_and_fallback_report(
    tmp_path: Path,
) -> None:
    from bigan.modeling import XGBoostV1Config, train_xgboost_v3
    from bigan.serving.readiness import run_xgboost_serving_readiness

    dataset_dir = tmp_path / "dataset"
    model_dir = tmp_path / "model"
    fallback_path = tmp_path / "baseline-model.json"
    rollback_path = tmp_path / "rollback.md"
    output_path = tmp_path / "serving_readiness.json"
    _write_dataset(dataset_dir)
    train_xgboost_v3(
        dataset_dir,
        model_dir,
        config=XGBoostV1Config(
            model_version="xgboost-v3",
            rounds_grid=(2,),
            learning_rate_grid=(0.3,),
            l2_penalty_grid=(5.0,),
            max_depth_grid=(2,),
            min_child_weight_grid=(1.0,),
            subsample_grid=(1.0,),
            colsample_bytree_grid=(1.0,),
        ),
    )
    fallback_path.write_text("{}", encoding="utf-8")
    rollback_path.write_text("# Rollback\n", encoding="utf-8")

    report = run_xgboost_serving_readiness(
        model_path=model_dir / "model.json",
        feature_schema_path=model_dir / "feature_schema.json",
        dataset_dir=dataset_dir,
        output_path=output_path,
        sample_size=6,
        batch_sizes=(6, 12),
        fallback_model_path=fallback_path,
        rollback_runbook_path=rollback_path,
    )

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written == report
    assert report["schema_version"] == "serving_readiness_v1"
    assert report["model_version"] == "xgboost-v3"
    assert report["ready"] is True
    assert report["p95_latency_ms"] >= 0.0
    assert report["error_rate"] == 0.0
    assert report["schema_validation"]["valid_input_accepted"] is True
    assert report["schema_validation"]["invalid_input_rejected"] is True
    assert report["schema_validation"]["silent_failure"] is False
    assert report["fallback"]["fallback_model_available"] is True
    assert report["fallback"]["rollback_runbook_available"] is True
    assert [row["batch_size"] for row in report["batch_throughput"]] == [6, 12]


def test_xgboost_serving_readiness_supports_v6_payload_model(tmp_path: Path) -> None:
    from bigan.modeling import XGBoostV6Config, train_xgboost_v6
    from bigan.serving.readiness import run_xgboost_serving_readiness

    dataset_dir = tmp_path / "v6-dataset"
    model_dir = tmp_path / "v6-model"
    fallback_path = tmp_path / "baseline-model.json"
    rollback_path = tmp_path / "rollback.md"
    output_path = tmp_path / "v6_serving_readiness.json"
    _write_v6_dataset(dataset_dir)
    train_xgboost_v6(
        dataset_dir,
        model_dir,
        config=XGBoostV6Config(
            rounds_grid=(2,),
            learning_rate_grid=(0.30,),
            l2_penalty_grid=(1.0,),
            max_depth_grid=(2,),
            min_child_weight_grid=(1.0,),
            subsample_grid=(1.0,),
            colsample_bytree_grid=(1.0,),
            temperature_grid=(1.0,),
            threshold_up_grid=(0.34,),
            neutral_cap_grid=(0.80,),
            volatility_threshold_grid=(0.10,),
            round_trip_cost=0.01,
            ev_margin=0.0,
            family_temperature_min_samples=3,
        ),
    )
    fallback_path.write_text("{}", encoding="utf-8")
    rollback_path.write_text("# Rollback\n", encoding="utf-8")

    report = run_xgboost_serving_readiness(
        model_path=model_dir / "model.json",
        feature_schema_path=model_dir / "feature_schema.json",
        dataset_dir=dataset_dir,
        output_path=output_path,
        sample_size=6,
        batch_sizes=(6, 12),
        fallback_model_path=fallback_path,
        rollback_runbook_path=rollback_path,
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == report
    assert report["model_version"] == "xgboost-v6"
    assert report["ready"] is True
    assert report["error_rate"] == 0.0
    assert [row["batch_size"] for row in report["batch_throughput"]] == [6, 12]
