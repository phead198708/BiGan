"""Contracts for xgboost-v7 settlement-EV artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _sample(
    feature_ts: int,
    *,
    side: str,
    ask: float,
    settlement: str,
    idx: int,
) -> dict:
    family = "BTC-15M" if idx % 2 == 0 else "ETH-15M"
    round_slug = f"{family.lower()}-updown-15m-{idx // 2}"
    canonical_symbol = f"{family}:{round_slug}:{side}"
    mid_price = ask if side == "UP" else 1.0 - ask
    return {
        "source": "polymarket",
        "source_symbol": f"tok-{idx}",
        "source_market": round_slug,
        "canonical_symbol": canonical_symbol,
        "symbol": canonical_symbol,
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
        "label_volatility_up": settlement == "UP" or idx % 3 == 0,
        "label_volatility_down": settlement == "DOWN" or idx % 4 == 0,
        "max_exit_gain_up": 0.28 if settlement == "UP" else 0.14,
        "max_exit_gain_down": 0.28 if settlement == "DOWN" else 0.14,
        "realized_return": 0.40 if settlement == "UP" else -0.40 if settlement == "DOWN" else 0.0,
        "completeness_score": 1.0,
        "data_gap_flag": False,
        "quality_filter_pass": True,
        "entry_ask_price": ask,
        "spread": 0.02 + abs(mid_price - 0.50) / 10,
        "mid_price": mid_price,
        "market_implied_prob": ask,
        "underlying_id": 0.0 if family == "BTC-15M" else 1.0,
        "horizon_minutes": 15.0,
        "liquidity_bucket": 1.0,
        "ret_15m": mid_price - 0.50,
        "minute_of_day": ((feature_ts // 60_000) % 1440) / 1439,
        "day_of_week": 2,
        "ret_30m": 2 * (mid_price - 0.50),
        "rv_30m": abs(mid_price - 0.50) + 0.05,
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
        "entry_ask_price",
        "ret_15m",
        *XGBOOST_V4_REQUIRED_MARKET_FEATURES,
        *XGBOOST_V4_REQUIRED_ADDED_FEATURES,
        *XGBOOST_V4_REQUIRED_TICK_FEATURES,
    ]
    specs = [
        ("UP", 0.34, "UP"),
        ("UP", 0.55, "NEUTRAL"),
        ("UP", 0.68, "DOWN"),
        ("DOWN", 0.36, "DOWN"),
        ("DOWN", 0.54, "NEUTRAL"),
        ("DOWN", 0.70, "UP"),
    ]
    train_rows = [
        _sample(idx * 60_000, side=side, ask=ask, settlement=settlement, idx=idx)
        for idx, (side, ask, settlement) in enumerate(specs * 2)
    ]
    val_rows = [
        _sample(1_000_000 + idx * 60_000, side=side, ask=ask, settlement=settlement, idx=30 + idx)
        for idx, (side, ask, settlement) in enumerate(specs)
    ]
    test_rows = [
        _sample(2_000_000 + idx * 60_000, side=side, ask=ask, settlement=settlement, idx=60 + idx)
        for idx, (side, ask, settlement) in enumerate(reversed(specs))
    ]
    _write_split(dataset_dir / "train.parquet", train_rows)
    _write_split(dataset_dir / "val.parquet", val_rows)
    _write_split(dataset_dir / "test.parquet", test_rows)
    (dataset_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": "bigan-training-v7-test",
                "feature_columns": feature_columns,
                "v5_feature_columns": feature_columns,
                "feature_versions": ["bigan-mvp-v1.0.0"],
                "label_versions": ["bigan-labels-v6.0.0"],
                "expected_sample_count_per_family": {"BTC-15M": 12, "ETH-15M": 12},
                "v6_label_diagnostics": {"phase4_capture_rows": 24},
                "rows_written": 24,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return feature_columns


def test_train_xgboost_v7_saves_settlement_ev_artifacts_and_payload(tmp_path: Path) -> None:
    from bigan.execution.signal_queue import append_prediction_rows_as_signal_jsonl
    from bigan.modeling import (
        XGBOOST_V7_MODEL_VERSION,
        XGBoostV7Config,
        generate_v7_prediction_rows,
        load_xgboost_v7_model,
        train_xgboost_v7,
    )

    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "xgb-v7"
    feature_columns = _write_dataset(dataset_dir)

    report = train_xgboost_v7(
        dataset_dir,
        output_dir,
        config=XGBoostV7Config(
            rounds_grid=(2,),
            learning_rate_grid=(0.30,),
            l2_penalty_grid=(1.0,),
            max_depth_grid=(2,),
            min_child_weight_grid=(1.0,),
            subsample_grid=(1.0,),
            colsample_bytree_grid=(1.0,),
            temperature_grid=(1.0, 1.5),
            settlement_threshold_grid=(0.50, 0.80),
            edge_threshold_grid=(0.0, 0.082),
            buy_slippage=0.02,
            ev_margin=0.01,
            family_temperature_min_samples=3,
        ),
    )

    model = load_xgboost_v7_model(output_dir / "model.json")
    payload = model.predict_payload(
        dict.fromkeys(feature_columns, 0.5)
        | {
            "canonical_symbol": "BTC-15M:btc-updown-15m-99:UP",
            "entry_ask_price": 0.52,
            "market_implied_prob": 0.52,
            "round_start_ts": 0,
            "feature_ts": 60_000,
            "round_end_ts": 960_000,
        }
    )
    wrapper = json.loads((output_dir / "model.json").read_text(encoding="utf-8"))

    assert report.model_version == XGBOOST_V7_MODEL_VERSION
    assert report.residual_metrics["test"]["label_formula"] == (
        "settlement_tradable_edge = win - market_implied_prob"
    )
    assert report.tradable_ev_metrics["test"]["v7_probability_ev_gate"]["metric_of_record"] == (
        "executable_one_way_settlement_pnl"
    )
    assert {item["bucket"] for item in report.tradable_ev_metrics["test"]["v7_probability_ev_gate"]["round_age_buckets"]} == {
        "0-180s",
        "180-360s",
        "360-540s",
        "540s+",
    }
    assert report.executor_contract["issue_101_guardrail"].startswith("Do not use live run PnL")
    assert "expected_edge_up" in wrapper["serving_payload"]
    assert "residual_expected_edge_down" in wrapper["serving_payload"]
    assert wrapper["compatibility"]["volatility"] == "not implemented in v7 settlement-EV artifact"
    assert (output_dir / "settlement_residual_model.json").exists()
    assert (output_dir / "executor_integration.md").exists()
    assert payload["model_version"] == XGBOOST_V7_MODEL_VERSION
    assert payload["settlement_residual"] is not None
    assert payload["entry_worst_price_up"] == pytest.approx(0.54)
    assert payload["expected_edge_up"] is not None
    assert payload["should_enter_settlement"] in {True, False, None}

    round_slug = "btc-updown-15m-1779774300"
    feature_row = dict.fromkeys(feature_columns, 0.5) | {
        "source": "polymarket",
        "source_symbol": "token-up",
        "source_market": round_slug,
        "canonical_symbol": f"BTC-15M:{round_slug}:UP",
        "symbol": f"BTC-15M:{round_slug}:UP",
        "feature_ts": 1_779_774_400_000,
        "feature_version": "bigan-mvp-v1.0.0",
        "entry_ask_price": 0.52,
        "market_implied_prob": 0.52,
        "round_start_ts": 1_779_773_500_000,
        "round_end_ts": 1_779_774_400_000 + 900_000,
    }
    prediction_rows = generate_v7_prediction_rows(
        feature_rows=[feature_row],
        model=model,
        ingest_ts=1_779_774_405_000,
    )
    prediction = prediction_rows[0]
    assert prediction["model_version"] == XGBOOST_V7_MODEL_VERSION
    assert prediction["p_vol_up"] is None
    assert prediction["expected_edge_up"] is not None
    assert prediction["selected_expected_edge"] is not None

    queue_path = tmp_path / "signals.jsonl"
    written = append_prediction_rows_as_signal_jsonl(
        queue_path,
        prediction_rows,
        model_version=XGBOOST_V7_MODEL_VERSION,
        bridged_at=1_779_774_410_000,
        token_ids_by_market_side={
            ("BTC-15M", round_slug, "UP"): "token-up",
            ("BTC-15M", round_slug, "DOWN"): "token-down",
        },
    )
    queue_payload = json.loads(queue_path.read_text(encoding="utf-8"))
    assert written == 1
    assert queue_payload["model_version"] == XGBOOST_V7_MODEL_VERSION
    assert queue_payload["outcome_side"] == prediction["selected_side"]
    assert queue_payload["canonical_symbol"].endswith(f":{queue_payload['outcome_side']}")
    assert queue_payload["selected_expected_edge"] == pytest.approx(
        prediction["selected_expected_edge"]
    )
    assert queue_payload["entry_worst_price"] is not None

    live_like_prediction = {
        **prediction,
        "p_up": 0.82,
        "p_down": 0.12,
        "prob_up_15m": 0.82,
        "selected_side": "UP",
        "selected_expected_edge": None,
        "entry_worst_price_up": None,
        "entry_worst_price_down": None,
        "expected_edge_up": None,
        "expected_edge_down": None,
        "residual_expected_edge_up": None,
        "residual_expected_edge_down": None,
        "market_implied_prob": 0.52,
    }
    live_like_queue_path = tmp_path / "signals-live-like.jsonl"
    written = append_prediction_rows_as_signal_jsonl(
        live_like_queue_path,
        [live_like_prediction],
        model_version=XGBOOST_V7_MODEL_VERSION,
        bridged_at=1_779_774_410_000,
        token_ids_by_market_side={
            ("BTC-15M", round_slug, "UP"): "token-up",
            ("BTC-15M", round_slug, "DOWN"): "token-down",
        },
    )
    live_like_queue_payload = json.loads(live_like_queue_path.read_text(encoding="utf-8"))
    assert written == 1
    assert live_like_queue_payload["outcome_side"] == "UP"
    assert live_like_queue_payload["canonical_symbol"] == f"BTC-15M:{round_slug}:UP"
    assert live_like_queue_payload["selected_expected_edge"] == pytest.approx(0.30)
    assert live_like_queue_payload["edge"] == pytest.approx(0.30)
    assert live_like_queue_payload["entry_worst_price"] is None


def test_xgboost_v7_settlement_ev_formulas() -> None:
    from bigan.modeling.xgboost_v7 import (
        _entry_worst_price,
        _settlement_realized_pnl,
        _settlement_residual_label,
        _tradable_ev_backtest,
        XGBoostV7Config,
    )

    winning_row = {
        "canonical_symbol": "BTC-15M:btc-updown-15m-1:UP",
        "label_settlement_3way": "UP",
        "market_implied_prob": 0.60,
        "entry_ask_price": 0.60,
    }
    losing_row = winning_row | {"label_settlement_3way": "DOWN"}
    cfg = XGBoostV7Config(
        rounds_grid=(1,),
        learning_rate_grid=(0.1,),
        l2_penalty_grid=(1.0,),
        max_depth_grid=(1,),
        min_child_weight_grid=(1.0,),
        subsample_grid=(1.0,),
        colsample_bytree_grid=(1.0,),
        temperature_grid=(1.0,),
        settlement_threshold_grid=(0.5,),
        edge_threshold_grid=(0.0,),
        buy_slippage=0.02,
        fee_bps=100.0,
    )

    assert _entry_worst_price(0.50, buy_slippage=0.02, fee_bps=100.0) == pytest.approx(0.525)
    assert _settlement_residual_label(winning_row) == pytest.approx(0.40)
    assert _settlement_residual_label(losing_row) == pytest.approx(-0.60)
    assert _settlement_realized_pnl(winning_row, cfg) == pytest.approx(0.374)

    ineligible_late_row = winning_row | {
        "feature_ts": 850_000,
        "round_start_ts": 0,
        "round_end_ts": 900_000,
    }
    summary = _tradable_ev_backtest(
        [ineligible_late_row],
        [
            {
                "p_up": 0.95,
                "p_down": 0.02,
                "entry_worst_price_up": 0.50,
                "entry_worst_price_down": 0.50,
                "expected_edge_up": 0.45,
                "expected_edge_down": -0.48,
            }
        ],
        {"settlement_threshold": 0.8, "edge_threshold": 0.0},
        cfg=cfg,
        probability_prefix="",
    )
    assert summary["trade_count"] == 0
    assert summary["candidate_round_count"] == 0


def test_xgboost_v7_gate_selection_prefers_stable_average_pnl() -> None:
    from bigan.modeling.xgboost_v7 import (
        _select_tradable_ev_rule,
        XGBoostV7Config,
    )

    def eligible_row(idx: int, *, label: str) -> dict:
        return {
            "canonical_symbol": f"BTC-15M:btc-updown-15m-{idx}:UP",
            "label_settlement_3way": label,
            "market_implied_prob": 0.50,
            "entry_ask_price": 0.50,
            "feature_ts": 60_000,
            "round_start_ts": 0,
            "round_end_ts": 900_000,
        }

    def payload(p_up: float, worst: float) -> dict:
        return {
            "p_up": p_up,
            "p_down": 1.0 - p_up,
            "entry_worst_price_up": worst,
            "entry_worst_price_down": 1.0 - worst,
            "expected_edge_up": p_up - worst,
            "expected_edge_down": (1.0 - p_up) - (1.0 - worst),
        }

    train_rows = [eligible_row(idx, label="UP") for idx in range(6)]
    train_payloads = [payload(0.80, 0.60) for _ in train_rows]
    val_rows = [eligible_row(idx, label="UP") for idx in range(10, 15)]
    val_payloads = [
        payload(0.80, 0.60),
        payload(0.80, 0.60),
        payload(0.73, 0.69),
        payload(0.73, 0.69),
        payload(0.73, 0.69),
    ]
    cfg = XGBoostV7Config(
        rounds_grid=(1,),
        learning_rate_grid=(0.1,),
        l2_penalty_grid=(1.0,),
        max_depth_grid=(1,),
        min_child_weight_grid=(1.0,),
        subsample_grid=(1.0,),
        colsample_bytree_grid=(1.0,),
        temperature_grid=(1.0,),
        settlement_threshold_grid=(0.70, 0.75),
        edge_threshold_grid=(0.04,),
        gate_selection_min_trades_per_split=2,
        gate_selection_min_avg_pnl=0.35,
    )

    selected = _select_tradable_ev_rule(
        {"train": train_rows, "val": val_rows},
        {"train": train_payloads, "val": val_payloads},
        cfg,
    )

    assert selected["settlement_threshold"] == pytest.approx(0.75)
    assert selected["edge_threshold"] == pytest.approx(0.04)
    assert selected["selection_method"] == "train_val_stability_min_avg_pnl"
    assert selected["selection_diagnostics"]["preferred"] is True
    assert selected["selection_diagnostics"]["strong_average_all_splits"] is True
    assert selected["validation"]["pnl"] < 1.76
