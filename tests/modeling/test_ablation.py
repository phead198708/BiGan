"""Feature ablation report contracts for issue #57."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _sample(feature_ts: int, mid_price: float, label: bool) -> dict:
    return {
        "source": "polymarket",
        "source_symbol": "tok-up",
        "source_market": "0xmkt",
        "canonical_symbol": "BTC-UP-15M",
        "symbol": "BTC-UP-15M",
        "feature_ts": feature_ts,
        "feature_version": "bigan-mvp-v1.0.0",
        "label_version": "bigan-labels-15m-v1.0.0",
        "target_ts": feature_ts + 900_000,
        "round_start_ts": feature_ts - 60_000,
        "round_end_ts": feature_ts + 900_000,
        "start_price": 100.0,
        "target_price": 101.0 if label else 99.0,
        "label_up_15m": label,
        "completeness_score": 1.0,
        "data_gap_flag": False,
        "quality_filter_pass": True,
        "spread": 0.02,
        "mid_price": mid_price,
        "market_implied_prob": mid_price,
        "ret_15m": mid_price - 0.50,
        "minute_of_day": (feature_ts // 60_000) % 1440,
        "day_of_week": 2,
        "ret_30m": 2 * (mid_price - 0.50),
        "rv_30m": abs(mid_price - 0.50),
        "aggressor_buy_ratio_1m": 0.75 if mid_price >= 0.50 else 0.25,
        "avg_trade_size_1m": 10.0 + mid_price,
        "tick_obi_l1": mid_price - 0.50,
        "tick_mid_price": mid_price,
    }


def _write_split(path: Path, rows: list[dict]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


def _write_dataset(dataset_dir: Path) -> None:
    dataset_dir.mkdir(parents=True)
    feature_columns = [
        "spread",
        "mid_price",
        "market_implied_prob",
        "ret_15m",
        "minute_of_day",
        "day_of_week",
        "ret_30m",
        "rv_30m",
        "aggressor_buy_ratio_1m",
        "avg_trade_size_1m",
        "tick_obi_l1",
        "tick_mid_price",
    ]
    splits = {
        "train": [0.38, 0.41, 0.44, 0.47, 0.53, 0.56, 0.59, 0.62],
        "val": [0.40, 0.46, 0.55, 0.61],
        "test": [0.39, 0.45, 0.57, 0.63],
    }
    offsets = {"train": 0, "val": 600_000, "test": 900_000}
    for split, values in splits.items():
        _write_split(
            dataset_dir / f"{split}.parquet",
            [
                _sample(offsets[split] + idx * 60_000, mid, label=mid >= 0.50)
                for idx, mid in enumerate(values)
            ],
        )
    (dataset_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": "bigan-training-15m-v1.0.0",
                "feature_columns": feature_columns,
                "feature_versions": ["bigan-mvp-v1.0.0"],
                "label_versions": ["bigan-labels-15m-v1.0.0"],
                "rows_written": 16,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_generate_feature_ablation_report_writes_json_and_markdown(tmp_path: Path) -> None:
    from bigan.modeling import (
        XGBoostV1Config,
        generate_feature_ablation_report,
        train_xgboost_v1,
    )

    dataset_dir = tmp_path / "dataset"
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "ablation"
    _write_dataset(dataset_dir)
    train_xgboost_v1(
        dataset_dir,
        model_dir,
        config=XGBoostV1Config(
            rounds_grid=(5,),
            learning_rate_grid=(0.60,),
            max_depth_grid=(2,),
        ),
    )

    report = generate_feature_ablation_report(
        model_dir / "model.json",
        dataset_dir,
        output_dir,
    )

    assert report.model_version == "xgboost-v1"
    assert report.split == "test"
    assert report.baseline_metrics["sample_count"] == 4
    assert report.replacement_strategy == "train_split_feature_mean"
    assert set(report.replacement_values) >= {"mid_price", "tick_obi_l1"}
    assert any(row.ablation_type == "feature" and row.name == "mid_price" for row in report.ablations)
    assert any(row.ablation_type == "group" and row.name == "tick_microstructure" for row in report.ablations)
    assert (output_dir / "feature_ablation.json").exists()
    assert (output_dir / "feature_ablation.md").exists()
    payload = json.loads((output_dir / "feature_ablation.json").read_text(encoding="utf-8"))
    assert payload["model_path"] == str(model_dir / "model.json")
    assert payload["ablations"][0]["deltas"].keys() == {
        "brier_score_increase",
        "roc_auc_drop",
    }


def test_feature_ablation_cli_writes_artifacts(tmp_path: Path) -> None:
    from bigan.ingestion.__main__ import feature_ablation_report_v1
    from bigan.modeling import XGBoostV1Config, train_xgboost_v1

    dataset_dir = tmp_path / "dataset"
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "ablation"
    _write_dataset(dataset_dir)
    train_xgboost_v1(
        dataset_dir,
        model_dir,
        config=XGBoostV1Config(
            rounds_grid=(5,),
            learning_rate_grid=(0.60,),
            max_depth_grid=(2,),
        ),
    )

    feature_ablation_report_v1(
        model_path=model_dir / "model.json",
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        calibration_path=None,
        split="test",
    )

    assert (output_dir / "feature_ablation.json").exists()
    assert (output_dir / "feature_ablation.md").exists()
