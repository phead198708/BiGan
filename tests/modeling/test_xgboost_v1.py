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
        "realized_return": 0.60 if label else -0.40,
        "completeness_score": 1.0,
        "data_gap_flag": False,
        "quality_filter_pass": True,
        "spread": 0.02 + abs(mid_price - 0.50) / 10,
        "mid_price": mid_price,
        "underlying_id": 0.0,
        "horizon_minutes": 15.0,
        "liquidity_bucket": 1.0,
        "ret_15m": mid_price - 0.50,
        "minute_of_day": ((feature_ts // 60_000) % 1440) / 1439,
        "day_of_week": 2,
        "ret_30m": 2 * (mid_price - 0.50),
        "rv_30m": abs(mid_price - 0.50),
        "aggressor_buy_ratio_1m": 0.75 if mid_price >= 0.50 else 0.25,
        "avg_trade_size_1m": 10.0 + mid_price,
        "tick_spread": 0.02 + abs(mid_price - 0.50) / 10,
        "tick_obi_l1": mid_price - 0.50,
        "tick_obi_l3": (mid_price - 0.50) / 2,
        "tick_mid_price": mid_price,
        "tick_price_velocity": mid_price - 0.50,
        "tick_trade_arrival_rate": 3.0 + abs(mid_price - 0.50),
    }


def _write_split(path: Path, rows: list[dict]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


def _write_dataset(dataset_dir: Path, *, feature_columns: list[str] | None = None) -> None:
    dataset_dir.mkdir(parents=True)
    feature_columns = feature_columns or ["spread", "mid_price", "ret_15m"]
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
                "feature_columns": feature_columns,
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
    assert (first_output / "family_metrics.json").exists()
    assert (first_output / "feature_importance.json").exists()
    assert (first_output / "feature_schema.json").exists()
    assert (first_output / "manifest.json").exists()
    feature_schema = json.loads((first_output / "feature_schema.json").read_text(encoding="utf-8"))
    assert feature_schema["feature_columns"] == ["spread", "mid_price", "ret_15m"]
    assert feature_schema["model_version"] == "xgboost-v1"
    assert feature_schema["schema_hash"]
    assert json.loads((first_output / "feature_importance.json").read_text(encoding="utf-8"))[
        0
    ] == first.feature_importance[0]
    family_metrics = json.loads((first_output / "family_metrics.json").read_text(encoding="utf-8"))
    assert family_metrics["test"]["BTC-15M"]["sample_count"] == 4
    assert first.family_metrics["test"]["BTC-15M"]["sample_count"] == 4
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


def test_xgboost_v1_default_search_space_is_not_too_shallow() -> None:
    from bigan.modeling import XGBoostV1Config

    config = XGBoostV1Config()

    assert config.rounds_grid == (100, 200, 300)
    assert config.learning_rate_grid == (0.01, 0.05, 0.10)
    assert config.l2_penalty_grid == (0.10, 1.0, 5.0)
    assert config.max_depth_grid == (3, 4, 5)
    assert config.min_child_weight_grid == (1.0,)
    assert config.subsample_grid == (0.70, 0.80, 1.0)
    assert config.colsample_bytree_grid == (0.70, 0.80, 1.0)


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


def test_train_xgboost_v2_saves_distinct_candidate_version(tmp_path: Path) -> None:
    from bigan.modeling import (
        XGBOOST_V2_MODEL_VERSION,
        load_xgboost_v1_model,
        train_xgboost_v2,
    )

    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "xgb-v2"
    _write_dataset(dataset_dir)

    report = train_xgboost_v2(dataset_dir, output_dir)
    model = load_xgboost_v1_model(output_dir / "model.json")
    feature_schema = json.loads((output_dir / "feature_schema.json").read_text(encoding="utf-8"))

    assert report.model_version == XGBOOST_V2_MODEL_VERSION
    assert model.model_version == XGBOOST_V2_MODEL_VERSION
    assert feature_schema["model_version"] == XGBOOST_V2_MODEL_VERSION
    assert json.loads((output_dir / "xgboost_config.json").read_text(encoding="utf-8"))[
        "model_version"
    ] == XGBOOST_V2_MODEL_VERSION
    config = json.loads((output_dir / "xgboost_config.json").read_text(encoding="utf-8"))
    assert config["rounds_grid"] == [100, 200, 300]
    assert config["max_depth_grid"] == [3, 4, 5]
    assert config["subsample_grid"] == [0.7, 0.8, 1.0]


def test_train_xgboost_v3_saves_conservative_candidate_version(tmp_path: Path) -> None:
    from bigan.modeling import (
        XGBOOST_V3_MODEL_VERSION,
        XGBoostV1Config,
        load_xgboost_v1_model,
        train_xgboost_v3,
    )

    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "xgb-v3"
    _write_dataset(dataset_dir)

    report = train_xgboost_v3(
        dataset_dir,
        output_dir,
        config=XGBoostV1Config(
            model_version=XGBOOST_V3_MODEL_VERSION,
            rounds_grid=(5,),
            learning_rate_grid=(0.30,),
            l2_penalty_grid=(5.0,),
            max_depth_grid=(3,),
            min_child_weight_grid=(2.0,),
            subsample_grid=(0.8,),
            colsample_bytree_grid=(0.8,),
        ),
    )
    model = load_xgboost_v1_model(output_dir / "model.json")
    feature_schema = json.loads((output_dir / "feature_schema.json").read_text(encoding="utf-8"))
    config = json.loads((output_dir / "xgboost_config.json").read_text(encoding="utf-8"))

    assert report.model_version == XGBOOST_V3_MODEL_VERSION
    assert model.model_version == XGBOOST_V3_MODEL_VERSION
    assert feature_schema["model_version"] == XGBOOST_V3_MODEL_VERSION
    assert config["model_version"] == XGBOOST_V3_MODEL_VERSION
    assert config["l2_penalty_grid"] == [5.0]
    assert config["max_depth_grid"] == [3]
    assert config["min_child_weight_grid"] == [2.0]


def test_train_xgboost_v4_saves_cv_and_light_ensemble_artifacts(tmp_path: Path) -> None:
    from bigan.modeling import (
        XGBOOST_V4_MODEL_VERSION,
        XGBOOST_V4_REQUIRED_ADDED_FEATURES,
        XGBOOST_V4_REQUIRED_MARKET_FEATURES,
        XGBOOST_V4_REQUIRED_TICK_FEATURES,
        XGBoostV1Config,
        load_xgboost_v1_model,
        train_xgboost_v4,
    )

    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "xgb-v4"
    feature_columns = [
        "spread",
        "mid_price",
        "ret_15m",
        *XGBOOST_V4_REQUIRED_MARKET_FEATURES,
        *XGBOOST_V4_REQUIRED_ADDED_FEATURES,
        *XGBOOST_V4_REQUIRED_TICK_FEATURES,
    ]
    _write_dataset(dataset_dir, feature_columns=feature_columns)

    report = train_xgboost_v4(
        dataset_dir,
        output_dir,
        config=XGBoostV1Config(
            model_version=XGBOOST_V4_MODEL_VERSION,
            rounds_grid=(5,),
            learning_rate_grid=(0.30,),
            l2_penalty_grid=(5.0,),
            max_depth_grid=(2,),
            min_child_weight_grid=(1.0,),
            subsample_grid=(1.0,),
            colsample_bytree_grid=(1.0,),
        ),
        ensemble_seeds=(0, 7),
    )
    model = load_xgboost_v1_model(output_dir / "model.json")
    wrapper = json.loads((output_dir / "model.json").read_text(encoding="utf-8"))
    ensemble_summary = json.loads(
        (output_dir / "ensemble_summary.json").read_text(encoding="utf-8")
    )
    feature_schema = json.loads((output_dir / "feature_schema.json").read_text(encoding="utf-8"))

    assert report.model_version == XGBOOST_V4_MODEL_VERSION
    assert report.cv_summary is not None
    assert report.ensemble is not None
    assert report.ensemble["model_version"] == XGBOOST_V4_MODEL_VERSION
    assert report.ensemble["member_count"] == 2
    assert report.ensemble["training_elapsed_seconds"] > 0.0
    assert report.ensemble["train_time_multiplier_estimate"] == 2
    assert report.ensemble["inference_eval_multiplier"] == 2
    assert ensemble_summary["train_time_multiplier_estimate"] == 2
    assert ensemble_summary["inference_eval_multiplier"] == 2
    assert report.ensemble["single_model_metrics"]["test"]["brier_score"] is not None
    assert report.ensemble["ensemble_metrics"]["test"]["brier_score"] is not None
    assert report.ensemble["ensemble_vs_single"]["acceptable"] is True
    assert report.metrics["test"]["pnl"] is not None
    assert report.metrics["test"]["trade_count"] is not None
    assert report.cv_summary["summary"]["pnl_mean"] is not None
    assert report.cv_summary["summary"]["pnl_std"] is not None
    assert wrapper["schema_version"] == "xgboost_ensemble_v1"
    assert wrapper["model_version"] == XGBOOST_V4_MODEL_VERSION
    assert wrapper["feature_columns"] == feature_columns
    assert [member["seed"] for member in wrapper["members"]] == [0, 7]
    assert (output_dir / "cv_summary.json").exists()
    assert (output_dir / "ensemble_summary.json").exists()
    assert (output_dir / "family_metrics.json").exists()
    assert (output_dir / "model_seed_0.json").exists()
    assert (output_dir / "model_seed_7.json").exists()
    assert model.model_version == XGBOOST_V4_MODEL_VERSION
    assert len(model.ensemble_boosters) == 2
    assert feature_schema["model_version"] == XGBOOST_V4_MODEL_VERSION
    assert set(XGBOOST_V4_REQUIRED_MARKET_FEATURES) <= set(model.feature_columns)
    assert set(XGBOOST_V4_REQUIRED_ADDED_FEATURES) <= set(model.feature_columns)
    assert set(XGBOOST_V4_REQUIRED_TICK_FEATURES) <= set(model.feature_columns)
    assert 0.0 <= model.predict_proba(_sample(999_000, 0.61, True)) <= 1.0


def test_train_xgboost_v5_saves_ensemble_with_v5_version(tmp_path: Path) -> None:
    from bigan.modeling import (
        XGBOOST_V4_REQUIRED_ADDED_FEATURES,
        XGBOOST_V4_REQUIRED_MARKET_FEATURES,
        XGBOOST_V4_REQUIRED_TICK_FEATURES,
        XGBOOST_V5_MODEL_VERSION,
        XGBoostV1Config,
        load_xgboost_v1_model,
        train_xgboost_v5,
    )

    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "xgb-v5"
    feature_columns = [
        "spread",
        "mid_price",
        "ret_15m",
        *XGBOOST_V4_REQUIRED_MARKET_FEATURES,
        *XGBOOST_V4_REQUIRED_ADDED_FEATURES,
        *XGBOOST_V4_REQUIRED_TICK_FEATURES,
    ]
    _write_dataset(dataset_dir, feature_columns=feature_columns)

    report = train_xgboost_v5(
        dataset_dir,
        output_dir,
        config=XGBoostV1Config(
            model_version=XGBOOST_V5_MODEL_VERSION,
            rounds_grid=(5,),
            learning_rate_grid=(0.30,),
            l2_penalty_grid=(5.0,),
            max_depth_grid=(2,),
            min_child_weight_grid=(1.0,),
            subsample_grid=(1.0,),
            colsample_bytree_grid=(1.0,),
        ),
        ensemble_seeds=(0, 7, 11),
    )
    wrapper = json.loads((output_dir / "model.json").read_text(encoding="utf-8"))
    model = load_xgboost_v1_model(output_dir / "model.json")

    assert report.model_version == XGBOOST_V5_MODEL_VERSION
    assert report.ensemble is not None
    assert report.ensemble["model_version"] == XGBOOST_V5_MODEL_VERSION
    assert report.ensemble["member_count"] == 3
    assert wrapper["model_version"] == XGBOOST_V5_MODEL_VERSION
    assert [member["seed"] for member in wrapper["members"]] == [0, 7, 11]
    assert model.model_version == XGBOOST_V5_MODEL_VERSION
    assert len(model.ensemble_boosters) == 3


def test_train_xgboost_v5_rejects_mismatched_model_version(tmp_path: Path) -> None:
    from bigan.modeling import (
        XGBOOST_V4_MODEL_VERSION,
        XGBoostV1Config,
        train_xgboost_v5,
    )

    with pytest.raises(ValueError, match="xgboost-v5 config must use model_version"):
        train_xgboost_v5(
            tmp_path / "dataset",
            tmp_path / "out",
            config=XGBoostV1Config(model_version=XGBOOST_V4_MODEL_VERSION),
        )


def test_fit_family_aware_calibration_writes_family_calibrators(tmp_path: Path) -> None:
    from bigan.modeling import (
        XGBOOST_V4_REQUIRED_ADDED_FEATURES,
        XGBOOST_V4_REQUIRED_MARKET_FEATURES,
        XGBOOST_V4_REQUIRED_TICK_FEATURES,
        XGBOOST_V5_MODEL_VERSION,
        XGBoostV1Config,
        fit_family_aware_calibration,
        train_xgboost_v5,
    )

    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "xgb-v5"
    calib_dir = tmp_path / "calib-family"
    feature_columns = [
        "spread",
        "mid_price",
        "ret_15m",
        *XGBOOST_V4_REQUIRED_MARKET_FEATURES,
        *XGBOOST_V4_REQUIRED_ADDED_FEATURES,
        *XGBOOST_V4_REQUIRED_TICK_FEATURES,
    ]
    _write_dataset(dataset_dir, feature_columns=feature_columns)
    train_xgboost_v5(
        dataset_dir,
        output_dir,
        config=XGBoostV1Config(
            model_version=XGBOOST_V5_MODEL_VERSION,
            rounds_grid=(5,),
            learning_rate_grid=(0.30,),
            l2_penalty_grid=(5.0,),
            max_depth_grid=(2,),
            min_child_weight_grid=(1.0,),
            subsample_grid=(1.0,),
            colsample_bytree_grid=(1.0,),
        ),
        ensemble_seeds=(0,),
    )

    report = fit_family_aware_calibration(
        output_dir / "model.json",
        dataset_dir,
        calib_dir,
    )

    calibration = json.loads((calib_dir / "calibration.json").read_text(encoding="utf-8"))
    assert report.model_version == XGBOOST_V5_MODEL_VERSION
    assert calibration["kind"] == "family_aware"
    assert (calib_dir / "calibration_report.json").exists()


def test_xgboost_v4_cv_keeps_duplicate_timestamps_in_same_fold() -> None:
    from bigan.modeling.xgboost_v1 import _time_series_cv_summary

    feature_columns = (
        "spread",
        "mid_price",
        "ret_15m",
        "minute_of_day",
        "day_of_week",
        "underlying_id",
        "horizon_minutes",
        "liquidity_bucket",
        "ret_30m",
        "rv_30m",
        "aggressor_buy_ratio_1m",
        "avg_trade_size_1m",
        "tick_spread",
        "tick_obi_l1",
        "tick_obi_l3",
        "tick_mid_price",
        "tick_price_velocity",
        "tick_trade_arrival_rate",
    )
    rows = [
        {
            **_sample(
                timestamp * 60_000,
                0.42 + ((timestamp + duplicate) % 8) * 0.03,
                label=((timestamp + duplicate) % 3 == 0),
            ),
            "source_symbol": f"tok-{timestamp}-{duplicate}",
        }
        for timestamp in range(20)
        for duplicate in range(2)
    ]

    summary = _time_series_cv_summary(
        rows,
        feature_columns,
        {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "eta": 0.3,
            "max_depth": 2,
            "lambda": 5.0,
            "verbosity": 0,
            "seed": 0,
            "rounds": 5,
        },
        model_version="xgboost-v4",
        max_folds=3,
    )

    assert summary["summary"]["fold_count"] == 3
    for fold in summary["folds"]:
        assert fold["train_end_ts"] < fold["val_start_ts"]
        assert fold["train_count"] % 2 == 0
        assert fold["val_count"] % 2 == 0
        assert fold["metrics"]["sample_count"] == fold["val_count"]


def test_train_xgboost_v4_rejects_dataset_missing_required_features(tmp_path: Path) -> None:
    from bigan.modeling import XGBOOST_V4_MODEL_VERSION, XGBoostV1Config, train_xgboost_v4

    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir)

    with pytest.raises(ValueError, match="missing required feature_columns") as exc:
        train_xgboost_v4(
            dataset_dir,
            tmp_path / "xgb-v4",
            config=XGBoostV1Config(
                model_version=XGBOOST_V4_MODEL_VERSION,
                rounds_grid=(1,),
                learning_rate_grid=(0.30,),
                l2_penalty_grid=(5.0,),
                max_depth_grid=(1,),
                min_child_weight_grid=(1.0,),
                subsample_grid=(1.0,),
                colsample_bytree_grid=(1.0,),
            ),
            ensemble_seeds=(0,),
        )
    message = str(exc.value)
    assert "market structure: underlying_id, horizon_minutes, liquidity_bucket" in message
    assert "issue #57 added: minute_of_day" in message
    assert "tick: tick_spread" in message


def test_train_xgboost_v4_rejects_dataset_missing_issue_57_added_features(
    tmp_path: Path,
) -> None:
    from bigan.modeling import (
        XGBOOST_V4_MODEL_VERSION,
        XGBOOST_V4_REQUIRED_MARKET_FEATURES,
        XGBOOST_V4_REQUIRED_TICK_FEATURES,
        XGBoostV1Config,
        train_xgboost_v4,
    )

    dataset_dir = tmp_path / "dataset"
    _write_dataset(
        dataset_dir,
        feature_columns=[
            "spread",
            "mid_price",
            "ret_15m",
            *XGBOOST_V4_REQUIRED_MARKET_FEATURES,
            *XGBOOST_V4_REQUIRED_TICK_FEATURES,
        ],
    )

    with pytest.raises(ValueError, match="issue #57 added") as exc:
        train_xgboost_v4(
            dataset_dir,
            tmp_path / "xgb-v4",
            config=XGBoostV1Config(
                model_version=XGBOOST_V4_MODEL_VERSION,
                rounds_grid=(1,),
                learning_rate_grid=(0.30,),
                l2_penalty_grid=(5.0,),
                max_depth_grid=(1,),
                min_child_weight_grid=(1.0,),
                subsample_grid=(1.0,),
                colsample_bytree_grid=(1.0,),
            ),
            ensemble_seeds=(0,),
        )
    assert "minute_of_day" in str(exc.value)


def test_train_xgboost_v4_rejects_all_null_required_feature_values(
    tmp_path: Path,
) -> None:
    from bigan.modeling import (
        XGBOOST_V4_MODEL_VERSION,
        XGBOOST_V4_REQUIRED_ADDED_FEATURES,
        XGBOOST_V4_REQUIRED_MARKET_FEATURES,
        XGBOOST_V4_REQUIRED_TICK_FEATURES,
        XGBoostV1Config,
        train_xgboost_v4,
    )

    dataset_dir = tmp_path / "dataset"
    _write_dataset(
        dataset_dir,
        feature_columns=[
            "spread",
            "mid_price",
            "ret_15m",
            *XGBOOST_V4_REQUIRED_MARKET_FEATURES,
            *XGBOOST_V4_REQUIRED_ADDED_FEATURES,
            *XGBOOST_V4_REQUIRED_TICK_FEATURES,
        ],
    )
    train_rows = pq.read_table(dataset_dir / "train.parquet").to_pylist()
    for row in train_rows:
        row["tick_spread"] = None
    _write_split(dataset_dir / "train.parquet", train_rows)

    with pytest.raises(ValueError, match="missing finite required feature values") as exc:
        train_xgboost_v4(
            dataset_dir,
            tmp_path / "xgb-v4",
            config=XGBoostV1Config(
                model_version=XGBOOST_V4_MODEL_VERSION,
                rounds_grid=(1,),
                learning_rate_grid=(0.30,),
                l2_penalty_grid=(5.0,),
                max_depth_grid=(1,),
                min_child_weight_grid=(1.0,),
                subsample_grid=(1.0,),
                colsample_bytree_grid=(1.0,),
            ),
            ensemble_seeds=(0,),
        )
    assert "train: tick_spread" in str(exc.value)


def test_train_xgboost_v4_rejects_all_null_required_validation_values(
    tmp_path: Path,
) -> None:
    from bigan.modeling import (
        XGBOOST_V4_MODEL_VERSION,
        XGBOOST_V4_REQUIRED_ADDED_FEATURES,
        XGBOOST_V4_REQUIRED_MARKET_FEATURES,
        XGBOOST_V4_REQUIRED_TICK_FEATURES,
        XGBoostV1Config,
        train_xgboost_v4,
    )

    dataset_dir = tmp_path / "dataset"
    _write_dataset(
        dataset_dir,
        feature_columns=[
            "spread",
            "mid_price",
            "ret_15m",
            *XGBOOST_V4_REQUIRED_MARKET_FEATURES,
            *XGBOOST_V4_REQUIRED_ADDED_FEATURES,
            *XGBOOST_V4_REQUIRED_TICK_FEATURES,
        ],
    )
    val_rows = pq.read_table(dataset_dir / "val.parquet").to_pylist()
    for row in val_rows:
        row["tick_obi_l1"] = None
    _write_split(dataset_dir / "val.parquet", val_rows)

    with pytest.raises(ValueError, match="missing finite required feature values") as exc:
        train_xgboost_v4(
            dataset_dir,
            tmp_path / "xgb-v4",
            config=XGBoostV1Config(
                model_version=XGBOOST_V4_MODEL_VERSION,
                rounds_grid=(1,),
                learning_rate_grid=(0.30,),
                l2_penalty_grid=(5.0,),
                max_depth_grid=(1,),
                min_child_weight_grid=(1.0,),
                subsample_grid=(1.0,),
                colsample_bytree_grid=(1.0,),
            ),
            ensemble_seeds=(0,),
        )
    assert "val: tick_obi_l1" in str(exc.value)


def test_train_xgboost_v1_rejects_empty_parameter_space(tmp_path: Path) -> None:
    from bigan.modeling import XGBoostV1Config

    with pytest.raises(ValueError, match="rounds_grid"):
        XGBoostV1Config(rounds_grid=())

    with pytest.raises(ValueError, match="min_child_weight_grid"):
        XGBoostV1Config(min_child_weight_grid=(0.0,))
