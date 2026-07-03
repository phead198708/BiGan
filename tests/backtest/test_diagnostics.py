"""Backtest diagnostic tests."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bigan.backtest import (
    PredictionSignal,
    run_grouped_threshold_backtest,
    run_model_threshold_backtest,
    run_oracle_label_sanity_backtest,
    run_prediction_threshold_backtest,
)


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _v6_backtest_sample(feature_ts: int, mid_price: float, settlement: str, idx: int) -> dict:
    up_vol = settlement == "UP" or idx % 3 == 0
    down_vol = settlement == "DOWN" or idx % 4 == 0
    return {
        "source": "polymarket",
        "source_symbol": f"tok-v6-down-{idx}",
        "source_market": "0xmkt",
        "canonical_symbol": "BTC-15M:btc-updown-15m-1000:DOWN",
        "symbol": "BTC-15M:btc-updown-15m-1000:DOWN",
        "feature_ts": feature_ts,
        "feature_version": "bigan-mvp-v1.0.0",
        "label_version": "bigan-labels-v6.0.0",
        "target_ts": feature_ts + 900_000,
        "round_start_ts": feature_ts - 60_000,
        "round_end_ts": feature_ts + 900_000,
        "start_price": 100.0,
        "target_price": 101.0 if settlement == "UP" else 99.0 if settlement == "DOWN" else 100.0,
        "label_up_15m": settlement == "UP",
        "label_profit_down_15m": settlement == "DOWN",
        "label_settlement_3way": settlement,
        "label_volatility_up": up_vol,
        "label_volatility_down": down_vol,
        "max_exit_gain_up": 0.24 + (0.06 if up_vol else 0.0),
        "max_exit_gain_down": 0.22 + (0.06 if down_vol else 0.0),
        "realized_return": 0.40 if settlement == "UP" else -0.40,
        "settlement_price": 1.0 if settlement == "DOWN" else 0.0,
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


def _write_v6_backtest_dataset(dataset_dir: Path) -> list[dict]:
    from bigan.modeling import (
        XGBOOST_V4_REQUIRED_ADDED_FEATURES,
        XGBOOST_V4_REQUIRED_MARKET_FEATURES,
        XGBOOST_V4_REQUIRED_TICK_FEATURES,
    )

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
        _v6_backtest_sample(idx * 60_000, mid_price, settlements[idx % len(settlements)], idx)
        for idx, mid_price in enumerate([0.34, 0.45, 0.64, 0.38, 0.50, 0.68])
    ]
    for split in ("train", "val", "test"):
        _write_parquet(dataset_dir / f"{split}.parquet", rows)
    (dataset_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": "bigan-training-v6-backtest-test",
                "feature_columns": feature_columns,
                "v5_feature_columns": feature_columns,
                "feature_versions": ["bigan-mvp-v1.0.0"],
                "label_versions": ["bigan-labels-v6.0.0"],
                "expected_sample_count_per_family": {"BTC-15M": 6},
                "v6_label_diagnostics": {"phase4_capture_rows": 6},
                "rows_written": 6,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return rows


def test_grouped_threshold_backtest_runs_each_source_symbol_independently(
    tmp_path: Path,
) -> None:
    report = run_grouped_threshold_backtest(
        signals=[
            PredictionSignal(ts=0, target_ts=1_000, prob_up_15m=0.70, source_symbol="tok-a"),
            PredictionSignal(ts=0, target_ts=1_000, prob_up_15m=0.80, source_symbol="tok-b"),
        ],
        quotes=[
            {"ts": 0, "source_symbol": "tok-a", "bid_price": 0.49, "ask_price": 0.50},
            {"ts": 1_000, "source_symbol": "tok-a", "bid_price": 0.55, "ask_price": 0.56},
            {"ts": 0, "source_symbol": "tok-b", "bid_price": 0.19, "ask_price": 0.20},
            {"ts": 1_000, "source_symbol": "tok-b", "bid_price": 0.30, "ask_price": 0.31},
        ],
        output_dir=tmp_path / "grouped",
        model_version="candidate-test",
        thresholds=(0.05,),
    )

    row = report.summary[0]
    assert row["trade_count"] == 2
    assert row["symbols_considered"] == 2
    assert row["symbols_with_quotes"] == 2
    assert row["net_pnl"] == pytest.approx(0.15)
    assert row["max_drawdown"] == pytest.approx(0.0)
    assert row["sharpe_ratio"] is not None
    assert row["trades_per_1000_signals"] == pytest.approx(1_000.0)
    assert row["top1_market_abs_net_pnl_share"] == pytest.approx(2 / 3)
    assert row["top5_market_abs_net_pnl_share"] == pytest.approx(1.0)
    assert json.loads((tmp_path / "grouped" / "summary.json").read_text(encoding="utf-8"))[
        0
    ]["trade_count"] == 2


def test_grouped_threshold_backtest_reports_drawdown_sortino_and_concentration(
    tmp_path: Path,
) -> None:
    report = run_grouped_threshold_backtest(
        signals=[
            PredictionSignal(ts=0, target_ts=1_000, prob_up_15m=0.80, source_symbol="tok-win"),
            PredictionSignal(ts=2_000, target_ts=3_000, prob_up_15m=0.80, source_symbol="tok-loss"),
        ],
        quotes=[
            {"ts": 0, "source_symbol": "tok-win", "bid_price": 0.19, "ask_price": 0.20},
            {"ts": 1_000, "source_symbol": "tok-win", "bid_price": 0.40, "ask_price": 0.41},
            {"ts": 2_000, "source_symbol": "tok-loss", "bid_price": 0.49, "ask_price": 0.50},
            {"ts": 3_000, "source_symbol": "tok-loss", "bid_price": 0.30, "ask_price": 0.31},
        ],
        output_dir=tmp_path / "grouped-risk",
        model_version="candidate-test",
        thresholds=(0.05,),
    )

    row = report.summary[0]
    assert row["trade_count"] == 2
    assert row["max_drawdown"] == pytest.approx(0.20)
    assert row["max_drawdown_pct"] == pytest.approx(1.0)
    assert row["sortino_ratio"] is not None
    assert row["turnover"] == pytest.approx(1.0)
    assert row["trades_per_day"] is not None
    assert row["concentration"]["top1_abs_net_pnl_share"] == pytest.approx(0.5)
    assert row["top5_market_abs_net_pnl_share"] == pytest.approx(1.0)


def test_oracle_label_sanity_flags_perfect_label_that_never_wins(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    rows = [
        {
            "feature_ts": idx * 10_000,
            "target_ts": idx * 10_000 + 1_000,
            "source": "polymarket",
            "source_symbol": "tok-up",
            "canonical_symbol": "BTC-15M:btc-updown-15m-test:UP",
            "label_up_15m": True,
        }
        for idx in range(3)
    ]
    for split, row in zip(("train", "val", "test"), rows, strict=True):
        _write_parquet(dataset_dir / f"{split}.parquet", [row])

    quote_rows = []
    for row in rows:
        quote_rows.extend(
            [
                {
                    "ts": row["feature_ts"],
                    "source_symbol": "tok-up",
                    "canonical_symbol": "BTC-15M:btc-updown-15m-test:UP",
                    "bid_price": 0.54,
                    "ask_price": 0.55,
                },
                {
                    "ts": row["target_ts"],
                    "source_symbol": "tok-up",
                    "canonical_symbol": "BTC-15M:btc-updown-15m-test:UP",
                    "bid_price": 0.45,
                    "ask_price": 0.46,
                },
            ]
        )
    warehouse_dir = tmp_path / "warehouse"
    _write_parquet(warehouse_dir / "raw_top_of_book" / "part-1.parquet", quote_rows)

    output_dir = tmp_path / "oracle"
    report = run_oracle_label_sanity_backtest(
        dataset_dir=dataset_dir,
        warehouse_dir=warehouse_dir,
        output_dir=output_dir,
        thresholds=(0.05,),
        use_label_target_ts=True,
    )

    assert report.issues == (
        "oracle_label_long_up_never_wins",
        "oracle_label_negative_net_pnl",
    )
    assert report.summary[0]["trade_count"] == 3
    assert report.summary[0]["win_rate"] == 0.0
    assert report.summary[0]["net_pnl"] < 0.0

    diagnostics = json.loads((output_dir / "diagnostics.json").read_text(encoding="utf-8"))
    trade = json.loads(
        (output_dir / "trade_log_sample_threshold_0_05.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert diagnostics["issues"] == list(report.issues)
    assert diagnostics["required_outcome_side"] == "UP"
    assert trade["exit_decision_ts"] == rows[0]["target_ts"]


def test_oracle_label_sanity_flags_missing_required_up_mapping(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    row = {
        "feature_ts": 0,
        "target_ts": 1_000,
        "source": "polymarket",
        "source_symbol": "tok-unknown",
        "canonical_symbol": None,
        "label_up_15m": True,
    }
    for split in ("train", "val", "test"):
        _write_parquet(dataset_dir / f"{split}.parquet", [row])
    warehouse_dir = tmp_path / "warehouse"
    _write_parquet(
        warehouse_dir / "raw_top_of_book" / "part-1.parquet",
        [
            {
                "ts": 0,
                "source_symbol": "tok-unknown",
                "canonical_symbol": None,
                "bid_price": 0.50,
                "ask_price": 0.52,
            }
        ],
    )

    report = run_oracle_label_sanity_backtest(
        dataset_dir=dataset_dir,
        warehouse_dir=warehouse_dir,
        output_dir=tmp_path / "oracle",
        thresholds=(0.05,),
    )

    assert "oracle_label_required_outcome_missing" in report.issues
    assert "oracle_quote_required_outcome_missing" in report.issues
    assert "oracle_label_no_trades" in report.issues
    assert report.summary[0]["signals_considered"] == 0


def test_prediction_threshold_backtest_filters_to_required_up_side(
    tmp_path: Path,
) -> None:
    warehouse_dir = tmp_path / "warehouse"
    _write_parquet(
        warehouse_dir / "predictions" / "part-1.parquet",
        [
            {
                "prediction_ts": 0,
                "source": "polymarket",
                "source_symbol": "tok-up",
                "canonical_symbol": "BTC-15M:btc-updown-15m-test:UP",
                "model_version": "xgboost-test",
                "prob_up_15m": 0.70,
                "market_implied_prob": 0.60,
            },
            {
                "prediction_ts": 0,
                "source": "polymarket",
                "source_symbol": "tok-down",
                "canonical_symbol": "BTC-15M:btc-updown-15m-test:DOWN",
                "model_version": "xgboost-test",
                "prob_up_15m": 0.90,
                "market_implied_prob": 0.84,
            },
        ],
    )
    _write_parquet(
        warehouse_dir / "labels_15m_v1" / "part-1.parquet",
        [
            {
                "feature_ts": 0,
                "target_ts": 1_000,
                "source": "polymarket",
                "source_symbol": "tok-up",
            },
            {
                "feature_ts": 0,
                "target_ts": 1_000,
                "source": "polymarket",
                "source_symbol": "tok-down",
            },
        ],
    )
    _write_parquet(
        warehouse_dir / "raw_top_of_book" / "part-1.parquet",
        [
            {
                "ts": 0,
                "source_symbol": "tok-up",
                "canonical_symbol": "BTC-15M:btc-updown-15m-test:UP",
                "bid_price": 0.49,
                "ask_price": 0.50,
            },
            {
                "ts": 1_000,
                "source_symbol": "tok-up",
                "canonical_symbol": "BTC-15M:btc-updown-15m-test:UP",
                "bid_price": 0.55,
                "ask_price": 0.56,
            },
            {
                "ts": 0,
                "source_symbol": "tok-down",
                "canonical_symbol": "BTC-15M:btc-updown-15m-test:DOWN",
                "bid_price": 0.20,
                "ask_price": 0.21,
            },
            {
                "ts": 1_000,
                "source_symbol": "tok-down",
                "canonical_symbol": "BTC-15M:btc-updown-15m-test:DOWN",
                "bid_price": 0.10,
                "ask_price": 0.11,
            },
        ],
    )

    report = run_prediction_threshold_backtest(
        warehouse_dir=warehouse_dir,
        output_dir=tmp_path / "prediction-backtest",
        model_version="xgboost-test",
        thresholds=(0.05,),
    )

    assert report.required_outcome_side == "UP"
    assert report.issues == ()
    assert report.summary[0]["signals_considered"] == 1
    assert report.summary[0]["trade_count"] == 1
    assert report.summary[0]["net_pnl"] == pytest.approx(0.05)
    trade = json.loads(
        (tmp_path / "prediction-backtest" / "trade_log_sample_threshold_0_05.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert trade["market_implied_prob"] == pytest.approx(0.60)
    assert trade["edge"] == pytest.approx(0.10)


def test_model_threshold_backtest_scores_saved_model_without_warehouse_predictions(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    rows = [
        {
            "feature_ts": 0,
            "target_ts": 1_000,
            "source": "polymarket",
            "source_symbol": "tok-up",
            "canonical_symbol": "BTC-15M:btc-updown-15m-test:UP",
            "label_up_15m": True,
            "mid_price": 0.70,
            "market_implied_prob": 0.50,
            "settlement_price": 0.85,
        },
        {
            "feature_ts": 2_000,
            "target_ts": 3_000,
            "source": "polymarket",
            "source_symbol": "tok-up",
            "canonical_symbol": "BTC-15M:btc-updown-15m-test:UP",
            "label_up_15m": False,
            "mid_price": 0.30,
            "market_implied_prob": 0.50,
            "settlement_price": 0.10,
        },
    ]
    for split in ("train", "val", "test"):
        _write_parquet(dataset_dir / f"{split}.parquet", rows)
    (dataset_dir / "manifest.json").write_text(
        json.dumps({"dataset_version": "dataset-v1"}, indent=2),
        encoding="utf-8",
    )
    model_path = tmp_path / "model.json"
    model_path.write_text(
        json.dumps(
            {
                "model_version": "logreg-test",
                "feature_columns": ["mid_price"],
                "coefficients": [10.0],
                "intercept": -5.0,
                "means": {"mid_price": 0.0},
                "scales": {"mid_price": 1.0},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    warehouse_dir = tmp_path / "warehouse"
    _write_parquet(
        warehouse_dir / "raw_top_of_book" / "part-1.parquet",
        [
            {
                "ts": 0,
                "source_symbol": "tok-up",
                "canonical_symbol": "BTC-15M:btc-updown-15m-test:UP",
                "bid_price": 0.49,
                "ask_price": 0.50,
            },
            {
                "ts": 2_000,
                "source_symbol": "tok-up",
                "canonical_symbol": "BTC-15M:btc-updown-15m-test:UP",
                "bid_price": 0.49,
                "ask_price": 0.50,
            },
        ],
    )

    report = run_model_threshold_backtest(
        model_path=model_path,
        dataset_dir=dataset_dir,
        warehouse_dir=warehouse_dir,
        output_dir=tmp_path / "backtest",
        thresholds=(0.30,),
        required_outcome_side="UP",
    )

    row = report.summary[0]
    assert report.model_version == "logreg-test"
    assert row["signals_considered"] == 6
    assert row["threshold_signals"] == 3
    assert row["trade_count"] == 1
    assert row["overlap_skipped"] == 2
    assert row["net_pnl"] == pytest.approx(0.35)
    diagnostics = json.loads((tmp_path / "backtest" / "diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["model_version"] == "logreg-test"
    assert diagnostics["metadata"]["backtest_kind"] == "direct_model"
    assert diagnostics["metadata"]["dataset_dir"] == str(dataset_dir)
    assert diagnostics["metadata"]["dataset_version"] == "dataset-v1"
    assert diagnostics["metadata"]["quote_filter"] == {
        "source_symbol_count": 1,
        "source_symbols_sample": ["tok-up"],
        "since_ts": 0,
        "until_ts": 3_000,
        "quote_request_count": 2,
    }


def test_model_threshold_backtest_restricts_to_market_families(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    rows = [
        {
            "feature_ts": 0,
            "target_ts": 1_000,
            "source": "polymarket",
            "source_symbol": "tok-15m",
            "canonical_symbol": "BTC-15M:btc-updown-15m-test:UP",
            "label_up_15m": True,
            "mid_price": 0.70,
            "market_implied_prob": 0.50,
            "settlement_price": 0.85,
        },
        {
            "feature_ts": 0,
            "target_ts": 1_000,
            "source": "polymarket",
            "source_symbol": "tok-5m",
            "canonical_symbol": "BTC-5M:btc-updown-5m-test:UP",
            "label_up_15m": True,
            "mid_price": 0.70,
            "market_implied_prob": 0.50,
            "settlement_price": 0.85,
        },
    ]
    for split in ("train", "val", "test"):
        _write_parquet(dataset_dir / f"{split}.parquet", rows)
    (dataset_dir / "manifest.json").write_text(
        json.dumps({"dataset_version": "dataset-v1"}, indent=2),
        encoding="utf-8",
    )
    model_path = tmp_path / "model.json"
    model_path.write_text(
        json.dumps(
            {
                "model_version": "logreg-test",
                "feature_columns": ["mid_price"],
                "coefficients": [10.0],
                "intercept": -5.0,
                "means": {"mid_price": 0.0},
                "scales": {"mid_price": 1.0},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    warehouse_dir = tmp_path / "warehouse"
    _write_parquet(
        warehouse_dir / "raw_top_of_book" / "part-1.parquet",
        [
            {
                "ts": 0,
                "source_symbol": "tok-15m",
                "canonical_symbol": "BTC-15M:btc-updown-15m-test:UP",
                "bid_price": 0.49,
                "ask_price": 0.50,
            },
            {
                "ts": 0,
                "source_symbol": "tok-5m",
                "canonical_symbol": "BTC-5M:btc-updown-5m-test:UP",
                "bid_price": 0.49,
                "ask_price": 0.50,
            },
        ],
    )

    run_model_threshold_backtest(
        model_path=model_path,
        dataset_dir=dataset_dir,
        warehouse_dir=warehouse_dir,
        output_dir=tmp_path / "backtest",
        thresholds=(0.10,),
        required_outcome_side="UP",
        market_families=frozenset({"BTC-15M"}),
    )

    diagnostics = json.loads(
        (tmp_path / "backtest" / "diagnostics.json").read_text(encoding="utf-8")
    )
    assert diagnostics["metadata"]["market_families"] == ["BTC-15M"]
    # Only the 15M symbol survives the family filter; its quote request count is 1.
    assert diagnostics["metadata"]["quote_filter"]["source_symbol_count"] == 1
    assert diagnostics["metadata"]["quote_filter"]["source_symbols_sample"] == ["tok-15m"]


def test_model_threshold_backtest_uses_down_token_probability_and_settlement(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    row = {
        "feature_ts": 0,
        "target_ts": 1_000,
        "source": "polymarket",
        "source_symbol": "tok-down",
        "canonical_symbol": "BTC-15M:btc-updown-15m-test:DOWN",
        "label_profit_down_15m": True,
        "mid_price": 0.0,
        "market_implied_prob": 0.30,
        "settlement_price": 1.0,
    }
    for split in ("train", "val", "test"):
        _write_parquet(dataset_dir / f"{split}.parquet", [row])
    (dataset_dir / "manifest.json").write_text(
        json.dumps({"dataset_version": "dataset-v1"}, indent=2),
        encoding="utf-8",
    )
    model_path = tmp_path / "model.json"
    model_path.write_text(
        json.dumps(
            {
                "model_version": "logreg-test",
                "feature_columns": ["mid_price"],
                "coefficients": [0.0],
                "intercept": -1.3862943611198906,
                "means": {"mid_price": 0.0},
                "scales": {"mid_price": 1.0},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    warehouse_dir = tmp_path / "warehouse"
    _write_parquet(
        warehouse_dir / "raw_top_of_book" / "part-1.parquet",
        [
            {
                "ts": 0,
                "source_symbol": "tok-down",
                "canonical_symbol": "BTC-15M:btc-updown-15m-test:DOWN",
                "bid_price": 0.29,
                "ask_price": 0.30,
            }
        ],
    )

    report = run_model_threshold_backtest(
        model_path=model_path,
        dataset_dir=dataset_dir,
        warehouse_dir=warehouse_dir,
        output_dir=tmp_path / "backtest",
        thresholds=(0.30,),
        required_outcome_side="DOWN",
    )

    row_summary = report.summary[0]
    assert report.required_outcome_side == "DOWN"
    assert report.issues == ()
    assert row_summary["signals_considered"] == 3
    assert row_summary["threshold_signals"] == 3
    assert row_summary["trade_count"] == 1
    assert row_summary["net_pnl"] == pytest.approx(0.70)
    trade = json.loads(
        (tmp_path / "backtest" / "trade_log_sample_threshold_0_3.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert trade["outcome_side"] == "DOWN"
    assert trade["prob_up_15m"] == pytest.approx(0.80)
    assert trade["market_implied_prob"] == pytest.approx(0.30)
    assert trade["edge"] == pytest.approx(0.50)
    diagnostics = json.loads((tmp_path / "backtest" / "diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["metadata"]["backtest_kind"] == "direct_model"
    assert diagnostics["metadata"]["dataset_version"] == "dataset-v1"


def test_model_threshold_backtest_uses_v6_explicit_down_probability(
    tmp_path: Path,
) -> None:
    from bigan.modeling import XGBoostV6Config, load_xgboost_v6_model, train_xgboost_v6

    dataset_dir = tmp_path / "v6-dataset"
    model_dir = tmp_path / "v6-model"
    rows = _write_v6_backtest_dataset(dataset_dir)
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
    warehouse_dir = tmp_path / "warehouse"
    quote_rows = []
    for row in rows:
        quote_rows.append(
            {
                "ts": row["feature_ts"],
                "source_symbol": row["source_symbol"],
                "canonical_symbol": row["canonical_symbol"],
                "bid_price": 0.001,
                "ask_price": 0.001,
            }
        )
    _write_parquet(warehouse_dir / "raw_top_of_book" / "part-1.parquet", quote_rows)

    report = run_model_threshold_backtest(
        model_path=model_dir / "model.json",
        dataset_dir=dataset_dir,
        warehouse_dir=warehouse_dir,
        output_dir=tmp_path / "backtest",
        thresholds=(0.0,),
        required_outcome_side="DOWN",
    )

    row_summary = report.summary[0]
    assert report.model_version == "xgboost-v6"
    assert report.required_outcome_side == "DOWN"
    assert report.issues == ()
    assert row_summary["trade_count"] > 0
    trade = json.loads(
        (tmp_path / "backtest" / "trade_log_sample_threshold_0_0.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    source_symbol = trade["source_symbol"]
    source_row = next(row for row in rows if row["source_symbol"] == source_symbol)
    payload = load_xgboost_v6_model(model_dir / "model.json").predict_payload(source_row)
    assert trade["outcome_side"] == "DOWN"
    assert trade["prob_up_15m"] == pytest.approx(float(payload["p_down"]))
    assert trade["prob_up_15m"] != pytest.approx(1.0 - float(payload["p_up"]))


def test_model_threshold_backtest_uses_v6_dataset_quote_proxy_without_warehouse(
    tmp_path: Path,
) -> None:
    from bigan.modeling import XGBoostV6Config, train_xgboost_v6

    dataset_dir = tmp_path / "v6-dataset"
    model_dir = tmp_path / "v6-model"
    _write_v6_backtest_dataset(dataset_dir)
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

    report = run_model_threshold_backtest(
        model_path=model_dir / "model.json",
        dataset_dir=dataset_dir,
        warehouse_dir=tmp_path / "empty-warehouse",
        output_dir=tmp_path / "backtest",
        thresholds=(0.0,),
        required_outcome_side="DOWN",
    )

    assert report.issues == ()
    assert report.summary[0]["trade_count"] > 0
    diagnostics = json.loads((tmp_path / "backtest" / "diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["metadata"]["quote_source"] == "dataset_market_implied_prob_proxy"
    assert diagnostics["metadata"]["quote_count"] == 6


def test_model_threshold_backtest_does_not_enter_after_settlement_window(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    row = {
        "feature_ts": 0,
        "target_ts": 1_000,
        "source": "polymarket",
        "source_symbol": "tok-up",
        "canonical_symbol": "BTC-15M:btc-updown-15m-test:UP",
        "label_up_15m": True,
        "mid_price": 0.70,
        "market_implied_prob": 0.50,
        "settlement_price": 0.85,
    }
    for split in ("train", "val", "test"):
        _write_parquet(dataset_dir / f"{split}.parquet", [row])
    model_path = tmp_path / "model.json"
    model_path.write_text(
        json.dumps(
            {
                "model_version": "logreg-test",
                "feature_columns": ["mid_price"],
                "coefficients": [10.0],
                "intercept": -5.0,
                "means": {"mid_price": 0.0},
                "scales": {"mid_price": 1.0},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    warehouse_dir = tmp_path / "warehouse"
    _write_parquet(
        warehouse_dir / "raw_top_of_book" / "part-1.parquet",
        [
            {
                "ts": 2_000,
                "source_symbol": "tok-up",
                "canonical_symbol": "BTC-15M:btc-updown-15m-test:UP",
                "bid_price": 0.49,
                "ask_price": 0.50,
            },
            {
                "ts": 0,
                "source_symbol": "tok-down",
                "canonical_symbol": "BTC-15M:btc-updown-15m-test:DOWN",
                "bid_price": 0.49,
                "ask_price": 0.50,
            },
        ],
    )

    report = run_model_threshold_backtest(
        model_path=model_path,
        dataset_dir=dataset_dir,
        warehouse_dir=warehouse_dir,
        output_dir=tmp_path / "backtest",
        thresholds=(0.30,),
        required_outcome_side="UP",
    )

    row_summary = report.summary[0]
    assert report.issues == ("model_quote_required_outcome_missing",)
    assert row_summary["signals_considered"] == 3
    assert row_summary["threshold_signals"] == 3
    assert row_summary["trade_count"] == 0
    assert row_summary["unfilled_signals"] == 3
    diagnostics = json.loads((tmp_path / "backtest" / "diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["metadata"]["quote_filter"] == {
        "source_symbol_count": 1,
        "source_symbols_sample": ["tok-up"],
        "since_ts": 0,
        "until_ts": 1_000,
        "quote_request_count": 1,
    }


def test_model_threshold_backtest_can_explicitly_use_dataset_quote_proxy(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    row = {
        "feature_ts": 0,
        "target_ts": 1_000,
        "source": "polymarket",
        "source_symbol": "tok-up",
        "canonical_symbol": "BTC-15M:btc-updown-15m-test:UP",
        "label_up_15m": True,
        "mid_price": 0.70,
        "market_implied_prob": 0.50,
        "settlement_price": 0.85,
    }
    for split in ("train", "val", "test"):
        _write_parquet(dataset_dir / f"{split}.parquet", [row])
    (dataset_dir / "manifest.json").write_text(
        json.dumps({"dataset_version": "dataset-v1"}, indent=2),
        encoding="utf-8",
    )
    model_path = tmp_path / "model.json"
    model_path.write_text(
        json.dumps(
            {
                "model_version": "logreg-test",
                "feature_columns": ["mid_price"],
                "coefficients": [10.0],
                "intercept": -5.0,
                "means": {"mid_price": 0.0},
                "scales": {"mid_price": 1.0},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report = run_model_threshold_backtest(
        model_path=model_path,
        dataset_dir=dataset_dir,
        warehouse_dir=tmp_path / "empty-warehouse",
        output_dir=tmp_path / "backtest",
        thresholds=(0.30,),
        required_outcome_side="UP",
        allow_dataset_quote_proxy=True,
    )

    row_summary = report.summary[0]
    assert report.issues == ()
    assert row_summary["trade_count"] == 1
    diagnostics = json.loads((tmp_path / "backtest" / "diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["metadata"]["quote_source"] == "dataset_market_implied_prob_proxy"
    assert diagnostics["metadata"]["quote_count"] == 1


def test_prediction_threshold_backtest_uses_label_settlement_when_available(
    tmp_path: Path,
) -> None:
    warehouse_dir = tmp_path / "warehouse"
    _write_parquet(
        warehouse_dir / "predictions" / "part-1.parquet",
        [
            {
                "prediction_ts": 0,
                "source": "polymarket",
                "source_symbol": "tok-up",
                "canonical_symbol": "BTC-15M:btc-updown-15m-test:UP",
                "model_version": "xgboost-test",
                "prob_up_15m": 0.70,
                "market_implied_prob": 0.50,
            }
        ],
    )
    _write_parquet(
        warehouse_dir / "labels_15m_v1" / "part-1.parquet",
        [
            {
                "feature_ts": 0,
                "target_ts": 1_000,
                "source": "polymarket",
                "source_symbol": "tok-up",
                "settlement_price": 1.0,
            }
        ],
    )
    _write_parquet(
        warehouse_dir / "raw_top_of_book" / "part-1.parquet",
        [
            {
                "ts": 0,
                "source_symbol": "tok-up",
                "canonical_symbol": "BTC-15M:btc-updown-15m-test:UP",
                "bid_price": 0.49,
                "ask_price": 0.50,
            }
        ],
    )

    report = run_prediction_threshold_backtest(
        warehouse_dir=warehouse_dir,
        output_dir=tmp_path / "prediction-settlement-backtest",
        model_version="xgboost-test",
        thresholds=(0.05,),
    )

    assert report.summary[0]["trade_count"] == 1
    assert report.summary[0]["net_pnl"] == pytest.approx(0.50)


def test_prediction_threshold_backtest_uses_down_token_probability_and_settlement(
    tmp_path: Path,
) -> None:
    warehouse_dir = tmp_path / "warehouse"
    _write_parquet(
        warehouse_dir / "predictions" / "part-1.parquet",
        [
            {
                "prediction_ts": 0,
                "source": "polymarket",
                "source_symbol": "tok-down",
                "canonical_symbol": "BTC-15M:btc-updown-15m-test:DOWN",
                "model_version": "xgboost-test",
                "prob_up_15m": 0.20,
                "market_implied_prob": 0.30,
            }
        ],
    )
    _write_parquet(
        warehouse_dir / "labels_15m_v1" / "part-1.parquet",
        [
            {
                "feature_ts": 0,
                "target_ts": 1_000,
                "source": "polymarket",
                "source_symbol": "tok-down",
                "settlement_price": 1.0,
                "label_profit_down_15m": True,
                "label_down_15m": True,
            }
        ],
    )
    _write_parquet(
        warehouse_dir / "raw_top_of_book" / "part-1.parquet",
        [
            {
                "ts": 0,
                "source_symbol": "tok-down",
                "canonical_symbol": "BTC-15M:btc-updown-15m-test:DOWN",
                "bid_price": 0.29,
                "ask_price": 0.30,
            }
        ],
    )

    report = run_prediction_threshold_backtest(
        warehouse_dir=warehouse_dir,
        output_dir=tmp_path / "prediction-down-backtest",
        model_version="xgboost-test",
        thresholds=(0.30,),
        required_outcome_side="DOWN",
    )

    assert report.required_outcome_side == "DOWN"
    assert report.issues == ()
    assert report.summary[0]["signals_considered"] == 1
    assert report.summary[0]["trade_count"] == 1
    assert report.summary[0]["net_pnl"] == pytest.approx(0.70)
    trade = json.loads(
        (tmp_path / "prediction-down-backtest" / "trade_log_sample_threshold_0_3.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert trade["prob_up_15m"] == pytest.approx(0.80)
    assert trade["market_implied_prob"] == pytest.approx(0.30)
    assert trade["edge"] == pytest.approx(0.50)
