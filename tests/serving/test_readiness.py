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
