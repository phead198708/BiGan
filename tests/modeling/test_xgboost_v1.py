"""TDD contract for issue #17 XGBoost v1 candidate training."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import xgboost as xgb


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
        "spread": 0.02 + abs(mid_price - 0.50) / 10,
        "mid_price": mid_price,
        "ret_15m": mid_price - 0.50,
    }


def _write_split(path: Path, rows: list[dict]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


def _write_dataset(dataset_dir: Path) -> None:
    dataset_dir.mkdir(parents=True)
    train_mid_prices = [0.38, 0.41, 0.44, 0.47, 0.53, 0.56, 0.59, 0.62]
    val_mid_prices = [0.40, 0.46, 0.55, 0.61]
    test_mid_prices = [0.39, 0.45, 0.57, 0.63]

    _write_split(
        dataset_dir / "train.parquet",
        [
            _sample(idx * 60_000, mid, label=mid >= 0.50)
            for idx, mid in enumerate(train_mid_prices)
        ],
    )
    _write_split(
        dataset_dir / "val.parquet",
        [
            _sample(600_000 + idx * 60_000, mid, label=mid >= 0.50)
            for idx, mid in enumerate(val_mid_prices)
        ],
    )
    _write_split(
        dataset_dir / "test.parquet",
        [
            _sample(900_000 + idx * 60_000, mid, label=mid >= 0.50)
            for idx, mid in enumerate(test_mid_prices)
        ],
    )
    (dataset_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": "bigan-training-15m-v1.0.0",
                "feature_columns": ["spread", "mid_price", "ret_15m"],
                "feature_versions": ["bigan-mvp-v1.0.0"],
                "label_versions": ["bigan-labels-15m-v1.0.0"],
                "rows_written": 16,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_train_xgboost_v1_tunes_saves_metrics_and_feature_importance(tmp_path: Path) -> None:
    from bigan.modeling import XGBoostV1Config, train_xgboost_v1

    dataset_dir = tmp_path / "dataset"
    first_output = tmp_path / "xgb-a"
    second_output = tmp_path / "xgb-b"
    _write_dataset(dataset_dir)

    config = XGBoostV1Config(
        rounds_grid=(2, 5),
        learning_rate_grid=(0.30, 0.60),
        l2_penalty_grid=(0.0, 1.0),
        max_depth_grid=(1, 2),
    )
    first = train_xgboost_v1(dataset_dir, first_output, config=config)
    second = train_xgboost_v1(dataset_dir, second_output, config=config)

    assert first.model_version == "xgboost-v1"
    assert first.best_params["rounds"] in (2, 5)
    assert first.best_params["max_depth"] in (1, 2)
    assert first.metrics["val"]["sample_count"] == 4
    assert first.metrics["test"]["roc_auc"] == pytest.approx(1.0)
    assert first.metrics["test"]["brier_score"] < 0.25
    assert first.feature_importance[0]["feature"] in {"mid_price", "ret_15m"}
    assert first.feature_importance[0]["gain"] > 0.0
    assert (first_output / "model.json").exists()
    assert (first_output / "xgboost_config.json").exists()
    assert (first_output / "metrics.json").exists()
    assert (first_output / "feature_importance.json").exists()
    assert (first_output / "manifest.json").exists()
    assert json.loads((first_output / "feature_importance.json").read_text(encoding="utf-8"))[
        0
    ] == first.feature_importance[0]
    assert json.loads((second_output / "model.json").read_text(encoding="utf-8")) == json.loads(
        (first_output / "model.json").read_text(encoding="utf-8")
    )
    assert second.metrics == first.metrics
    assert "logistic-loss-gradient-boosted-stumps" not in (
        first_output / "model.json"
    ).read_text(encoding="utf-8")

    booster = xgb.Booster()
    booster.load_model(str(first_output / "model.json"))
    assert booster.attr("model_version") == "xgboost-v1"
    assert json.loads(booster.attr("feature_columns") or "[]") == [
        "spread",
        "mid_price",
        "ret_15m",
    ]


def test_loaded_xgboost_v1_predicts_and_explains_top_features(tmp_path: Path) -> None:
    from bigan.modeling import XGBoostV1Config, load_xgboost_v1_model, train_xgboost_v1

    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "xgb"
    _write_dataset(dataset_dir)
    train_xgboost_v1(
        dataset_dir,
        output_dir,
        config=XGBoostV1Config(
            rounds_grid=(5,),
            learning_rate_grid=(0.60,),
            max_depth_grid=(2,),
        ),
    )

    model = load_xgboost_v1_model(output_dir / "model.json")
    low_prob, high_prob = model.predict_proba_many(
        [
            {"spread": 0.03, "mid_price": 0.42, "ret_15m": -0.08},
            {"spread": 0.03, "mid_price": 0.61, "ret_15m": 0.11},
        ]
    )
    top_features = model.top_feature_contributions(
        {"spread": 0.03, "mid_price": 0.61, "ret_15m": 0.11},
        limit=2,
    )

    assert model.feature_columns == ("spread", "mid_price", "ret_15m")
    assert 0.0 <= low_prob <= 1.0
    assert 0.0 <= high_prob <= 1.0
    assert high_prob > low_prob
    assert top_features
    assert top_features[0]["abs_contribution"] >= top_features[-1]["abs_contribution"]


def test_train_xgboost_v1_rejects_empty_parameter_space(tmp_path: Path) -> None:
    from bigan.modeling import XGBoostV1Config

    with pytest.raises(ValueError, match="rounds_grid"):
        XGBoostV1Config(rounds_grid=())
