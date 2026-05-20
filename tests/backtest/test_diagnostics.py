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
    run_oracle_label_sanity_backtest,
    run_prediction_threshold_backtest,
)


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


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
    assert json.loads((tmp_path / "grouped" / "summary.json").read_text(encoding="utf-8"))[
        0
    ]["trade_count"] == 2


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
