"""Contracts for xgboost-v6 settlement and volatility heads."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _sample(feature_ts: int, mid_price: float, settlement: str, idx: int) -> dict:
    up_vol = settlement == "UP" or idx % 3 == 0
    down_vol = settlement == "DOWN" or idx % 4 == 0
    return {
        "source": "polymarket",
        "source_symbol": f"tok-{idx}",
        "source_market": "0xmkt",
        "canonical_symbol": "BTC-UPDOWN-15M" if idx % 2 == 0 else "ETH-UPDOWN-15M",
        "symbol": "BTC-UPDOWN-15M" if idx % 2 == 0 else "ETH-UPDOWN-15M",
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
        "underlying_id": 0.0 if idx % 2 == 0 else 1.0,
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


def _write_split(path: Path, rows: list[dict]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


def _write_dataset(dataset_dir: Path) -> list[str]:
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
    train_rows = [
        _sample(idx * 60_000, mid_price, settlements[idx % len(settlements)], idx)
        for idx, mid_price in enumerate([0.34, 0.45, 0.64, 0.38, 0.50, 0.68, 0.32, 0.48, 0.70])
    ]
    val_rows = [
        _sample(900_000 + idx * 60_000, mid_price, settlements[idx % len(settlements)], 20 + idx)
        for idx, mid_price in enumerate([0.36, 0.49, 0.66, 0.42, 0.52, 0.72])
    ]
    test_rows = [
        _sample(1_500_000 + idx * 60_000, mid_price, settlements[idx % len(settlements)], 40 + idx)
        for idx, mid_price in enumerate([0.35, 0.47, 0.67, 0.39, 0.51, 0.73])
    ]
    _write_split(dataset_dir / "train.parquet", train_rows)
    _write_split(dataset_dir / "val.parquet", val_rows)
    _write_split(dataset_dir / "test.parquet", test_rows)
    (dataset_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": "bigan-training-v6-test",
                "feature_columns": feature_columns,
                "v5_feature_columns": feature_columns,
                "feature_versions": ["bigan-mvp-v1.0.0"],
                "label_versions": ["bigan-labels-v6.0.0"],
                "expected_sample_count_per_family": {"BTC-15M": 11, "ETH-15M": 10},
                "v6_label_diagnostics": {"phase4_capture_rows": 21},
                "rows_written": 21,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return feature_columns


def test_train_xgboost_v6_saves_multihead_artifacts_and_payload(tmp_path: Path) -> None:
    from bigan.modeling import (
        XGBOOST_V6_MODEL_VERSION,
        XGBoostV6Config,
        load_xgboost_v6_model,
        train_xgboost_v6,
    )

    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "xgb-v6"
    feature_columns = _write_dataset(dataset_dir)

    report = train_xgboost_v6(
        dataset_dir,
        output_dir,
        config=XGBoostV6Config(
            rounds_grid=(2,),
            learning_rate_grid=(0.30,),
            l2_penalty_grid=(1.0,),
            max_depth_grid=(2,),
            min_child_weight_grid=(1.0,),
            subsample_grid=(1.0,),
            colsample_bytree_grid=(1.0,),
            temperature_grid=(1.0, 1.5),
            threshold_up_grid=(0.34,),
            neutral_cap_grid=(0.80,),
            volatility_threshold_grid=(0.10,),
            round_trip_cost=0.01,
            ev_margin=0.0,
            family_temperature_min_samples=3,
        ),
    )

    model = load_xgboost_v6_model(output_dir / "model.json")
    payload = model.predict_payload(
        dict.fromkeys(feature_columns, 0.5)
        | {
            "canonical_symbol": "BTC-UPDOWN-15M",
            "mid_price": 0.62,
            "market_implied_prob": 0.62,
        }
    )
    wrapper = json.loads((output_dir / "model.json").read_text(encoding="utf-8"))

    assert report.model_version == XGBOOST_V6_MODEL_VERSION
    assert report.metrics["test"]["per_class_ece"].keys() == {"UP", "DOWN", "NEUTRAL"}
    assert report.volatility_metrics["test"]["up"]["trivial_baseline"]["sample_count"] == 6
    assert report.volatility_metrics["test"]["down"]["bucket_hit_rate"]
    assert report.cost_adjusted_backtest["test"]["metric_of_record"] == (
        "cost_adjusted_account_cashflow_proxy_pnl"
    )
    assert report.v5_comparison["available"] is True
    assert report.feature_parity["empty"] is True
    assert report.coverage["depends_on_issue_91_v6_label_coverage"] is True
    assert wrapper["serving_payload"] == [
        "p_up",
        "p_down",
        "p_neutral",
        "p_vol_up",
        "p_vol_down",
        "model_version",
    ]
    assert wrapper["compatibility"]["down_probability"] == (
        "must be read from p_down; never derive from 1 - p_up"
    )
    assert (output_dir / "executor_integration.md").exists()
    assert payload["model_version"] == XGBOOST_V6_MODEL_VERSION
    assert float(payload["p_down"]) != pytest.approx(1.0 - float(payload["p_up"]))
    assert (
        float(payload["p_up"]) + float(payload["p_down"]) + float(payload["p_neutral"])
    ) == pytest.approx(1.0)
    assert 0.0 <= float(payload["p_vol_up"]) <= 1.0
    assert 0.0 <= float(payload["p_vol_down"]) <= 1.0


def test_volatility_bucket_hit_rate_keeps_high_bucket_exclusive() -> None:
    from bigan.modeling.xgboost_v6 import _bucket_hit_rate

    buckets = _bucket_hit_rate(
        labels=[0, 1, 1, 0, 1],
        probabilities=[0.49, 0.50, 0.69, 0.70, 1.0],
    )

    assert buckets["0.00-0.50"]["sample_count"] == 1
    assert buckets["0.50-0.60"]["sample_count"] == 1
    assert buckets["0.60-0.70"]["sample_count"] == 1
    assert buckets["0.70-1.00"]["sample_count"] == 2
    assert buckets["0.70-1.00"]["hit_rate"] == pytest.approx(0.5)
