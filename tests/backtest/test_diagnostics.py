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
