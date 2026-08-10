from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.challenge_model_15m_training import (
    BASE_FEATURE_NAMES,
    GLOBAL_RAW_DEPENDENCIES,
    SIDE_RAW_SUFFIXES,
    _side_raw_name,
    side_symmetric_features,
)
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.corpus.contracts import (
    BinanceBTCCandle,
    PolymarketChainlinkPrice,
    PolymarketCorpusBookSnapshot,
    PolymarketCorpusMarket,
)
from bigan.v8.polymarket.cost_aware_residual_v4_stacking import (
    DEFAULT_OUTPUT_DIR as V4_CHALLENGER_OUTPUT_DIR,
)
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.pooled_residual_runtime import (
    ACTIONS,
    BUNDLE_SCHEMA_VERSION,
    CALIBRATION_ELIGIBILITY_SCHEMA_VERSION,
    PooledResidualRuntimeError,
    build_pooled_residual_feature_row_as_of,
    load_pooled_residual_runtime,
)
from bigan.v8.polymarket.regime_adaptive_candidate_evaluation import FEATURE_NAMES


def test_repository_local_ubj_bundle_loads_and_matches_offline_prediction(
    tmp_path: Path,
) -> None:
    runtime, booster = _runtime_fixture(tmp_path)
    feature_row = _feature_row()
    matrix = np.asarray(
        [
            list(side_symmetric_features(feature_row, side).values())
            for side in ("UP", "DOWN")
        ],
        dtype=np.float64,
    )
    expected = booster.predict(
        xgb.DMatrix(matrix, feature_names=list(FEATURE_NAMES), missing=np.nan)
    )
    result = runtime.score_feature_row(feature_row, observed_at_ts=1_000_500)
    assert result["model_scored"] is True
    assert result["fail_closed"] is False
    assert result["action_values"]["BUY_UP_HOLD"] == pytest.approx(expected[0])
    assert result["action_values"]["BUY_DOWN_HOLD"] == pytest.approx(expected[1])
    expected_action = (
        "BUY_UP_HOLD"
        if expected[0] >= expected[1] and expected[0] > 0.0
        else "BUY_DOWN_HOLD"
        if expected[1] > 0.0
        else "NO_TRADE"
    )
    assert result["selected_action"] == expected_action
    assert tuple(result["action_values"]) == ACTIONS
    assert result["offline_readiness_only"] is True
    assert result["live_shadow_authorized"] is False
    assert result["paper_or_live_execution_authorized"] is False
    assert result["polymarket_write_authorized"] is False
    assert result["wallet_signing_authorized"] is False
    assert result["capital_at_risk"] is False
    assert all(value is False for value in result["safety"].values())


def test_manifest_resolution_is_repository_relative_not_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _bundle_fixture(tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    runtime = load_pooled_residual_runtime(
        manifest_path=fixture["manifest_path"].relative_to(tmp_path),
        expected_manifest_sha256=fixture["manifest_sha"],
        repository_root=tmp_path,
    )
    assert runtime.manifest_sha256 == fixture["manifest_sha"]


def test_standard_corpus_feature_row_needs_no_start_or_end_extension(
    tmp_path: Path,
) -> None:
    runtime, _ = _runtime_fixture(tmp_path)
    row = _feature_row()
    row.pop("market_start_ts")
    row.pop("market_end_ts")
    result = runtime.score_feature_row(row, observed_at_ts=1_000_500)
    assert result["model_scored"] is True
    assert result["fail_closed"] is False


def test_as_of_stream_builder_ignores_future_inputs_and_scores_exact_row(
    tmp_path: Path,
) -> None:
    runtime, _ = _runtime_fixture(tmp_path)
    market, books, candles, chainlink, decision_ts = _causal_stream_fixture()
    without_future = build_pooled_residual_feature_row_as_of(
        market=market,
        book_snapshots=books[:-2],
        trades=(),
        btc_candles=candles[:-1],
        chainlink_prices=chainlink[:-1],
        decision_ts=decision_ts,
    )
    with_future = build_pooled_residual_feature_row_as_of(
        market=market,
        book_snapshots=books,
        trades=(),
        btc_candles=candles,
        chainlink_prices=chainlink,
        decision_ts=decision_ts,
    )
    assert with_future == without_future
    assert with_future["max_input_ts"] <= decision_ts
    assert with_future["available_at_ts"] <= decision_ts
    assert with_future["feature_cutoff_ts"] <= decision_ts
    result = runtime.score_feature_row(with_future, observed_at_ts=decision_ts + 500)
    assert result["model_scored"] is True
    assert result["fail_closed"] is False


@pytest.mark.parametrize(
    "mutation",
    ["manifest_sha", "model_sha", "path_escape", "failed_candidate"],
)
def test_runtime_artifact_or_candidate_mismatch_fails_closed(
    tmp_path: Path, mutation: str
) -> None:
    fixture = _bundle_fixture(tmp_path)
    manifest = json.loads(fixture["manifest_path"].read_text(encoding="utf-8"))
    expected_manifest_sha = fixture["manifest_sha"]
    if mutation == "manifest_sha":
        expected_manifest_sha = "0" * 64
    elif mutation == "model_sha":
        manifest["model"]["sha256"] = "0" * 64
    elif mutation == "path_escape":
        manifest["model"]["path"] = "../escaped.ubj"
    else:
        freeze = json.loads(fixture["freeze_path"].read_text(encoding="utf-8"))
        freeze["all_gates_passed"] = False
        freeze["candidate_freeze_allowed"] = False
        _write_json(fixture["freeze_path"], freeze)
        manifest["candidate_freeze"]["sha256"] = sha256_file(fixture["freeze_path"])
    if mutation != "manifest_sha":
        _write_json(fixture["manifest_path"], manifest)
        expected_manifest_sha = sha256_file(fixture["manifest_path"])
    with pytest.raises(PooledResidualRuntimeError):
        load_pooled_residual_runtime(
            manifest_path=fixture["manifest_path"],
            expected_manifest_sha256=expected_manifest_sha,
            repository_root=tmp_path,
        )


def test_calibration_eligibility_sha_or_semantics_drift_fails_closed(
    tmp_path: Path,
) -> None:
    fixture = _bundle_fixture(tmp_path)
    eligibility = json.loads(
        fixture["eligibility_path"].read_text(encoding="utf-8")
    )
    eligibility["fixed_acceptance_threshold"] = 0.01
    _write_json(fixture["eligibility_path"], eligibility)
    manifest = json.loads(fixture["manifest_path"].read_text(encoding="utf-8"))
    manifest["calibration_eligibility"] = _descriptor(
        fixture["eligibility_path"], tmp_path
    )
    _write_json(fixture["manifest_path"], manifest)
    with pytest.raises(PooledResidualRuntimeError, match="calibration eligibility"):
        load_pooled_residual_runtime(
            manifest_path=fixture["manifest_path"],
            expected_manifest_sha256=sha256_file(fixture["manifest_path"]),
            repository_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("wrong_family", "not BTC 15m"),
        ("wrong_horizon", "not 900 seconds"),
        ("future_input", "causality violation"),
        ("stale_decision", "decision input is stale"),
        ("stale_source", "source input is stale"),
        ("missing_field", "feature provenance missing"),
    ],
)
def test_invalid_or_stale_input_always_returns_no_trade(
    tmp_path: Path, mutation: str, reason: str
) -> None:
    runtime, _ = _runtime_fixture(tmp_path)
    row = _feature_row()
    observed_at = 1_000_500
    if mutation == "wrong_family":
        row["market_family"] = "btc_updown_5m"
    elif mutation == "wrong_horizon":
        row["horizon_ms"] = 300_000
    elif mutation == "future_input":
        row["max_input_ts"] = row["decision_ts"] + 1
    elif mutation == "stale_decision":
        observed_at = row["decision_ts"] + 5_001
    elif mutation == "stale_source":
        row["max_input_ts"] = row["decision_ts"] - 5_001
    else:
        row["feature_provenance"].pop("up_ask")
    result = runtime.score_feature_row(row, observed_at_ts=observed_at)
    assert result["selected_action"] == "NO_TRADE"
    assert result["model_scored"] is False
    assert result["fail_closed"] is True
    assert reason in result["fail_closed_reasons"][0]


def test_native_nan_is_preserved_and_never_converted_to_zero(tmp_path: Path) -> None:
    runtime, _ = _runtime_fixture(tmp_path)
    row = _feature_row()
    row["features"]["recent_up_trade_volume"] = None
    row["features"]["recent_down_trade_volume"] = None
    up = side_symmetric_features(row, "UP")
    assert np.isnan(up["selected_recent_trade_volume"])
    assert up["selected_recent_trade_volume__missing"] == 1.0
    result = runtime.score_feature_row(row, observed_at_ts=1_000_500)
    assert result["model_scored"] is True
    assert result["fail_closed"] is False


def test_runtime_rejects_ubj_feature_order_drift(tmp_path: Path) -> None:
    fixture = _bundle_fixture(tmp_path, reversed_features=True)
    with pytest.raises(PooledResidualRuntimeError, match="feature names/order"):
        load_pooled_residual_runtime(
            manifest_path=fixture["manifest_path"],
            expected_manifest_sha256=fixture["manifest_sha"],
            repository_root=tmp_path,
        )


def test_runtime_rejects_real_v4_terminal_failed_candidate(tmp_path: Path) -> None:
    fixture = _bundle_fixture(tmp_path)
    failed = json.loads(
        (
            V4_CHALLENGER_OUTPUT_DIR / "residual_v4_stacking_oof_report.json"
        ).read_text(encoding="utf-8")
    )
    _write_json(fixture["freeze_path"], failed)
    manifest = json.loads(fixture["manifest_path"].read_text(encoding="utf-8"))
    manifest["candidate_id"] = "residual-v4-challenger-slot-002"
    manifest["lineage_id"] = failed["lineage_id"]
    eligibility = json.loads(
        fixture["eligibility_path"].read_text(encoding="utf-8")
    )
    eligibility["candidate_id"] = manifest["candidate_id"]
    eligibility["lineage_id"] = manifest["lineage_id"]
    _write_json(fixture["eligibility_path"], eligibility)
    manifest["calibration_eligibility"] = _descriptor(
        fixture["eligibility_path"], tmp_path
    )
    manifest["candidate_freeze"] = _descriptor(fixture["freeze_path"], tmp_path)
    _write_json(fixture["manifest_path"], manifest)
    with pytest.raises(PooledResidualRuntimeError, match="not OOF-gated and frozen"):
        load_pooled_residual_runtime(
            manifest_path=fixture["manifest_path"],
            expected_manifest_sha256=sha256_file(fixture["manifest_path"]),
            repository_root=tmp_path,
        )


def _runtime_fixture(tmp_path: Path) -> tuple[object, xgb.Booster]:
    fixture = _bundle_fixture(tmp_path)
    runtime = load_pooled_residual_runtime(
        manifest_path=fixture["manifest_path"],
        expected_manifest_sha256=fixture["manifest_sha"],
        repository_root=tmp_path,
    )
    return runtime, fixture["booster"]


def _bundle_fixture(tmp_path: Path, *, reversed_features: bool = False) -> dict:
    bundle = tmp_path / "runtime_bundle"
    bundle.mkdir()
    names = tuple(reversed(FEATURE_NAMES)) if reversed_features else FEATURE_NAMES
    training = np.asarray(
        [np.linspace(-1.0, 1.0, len(names)), np.linspace(1.0, -1.0, len(names))],
        dtype=np.float64,
    )
    booster = xgb.train(
        {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "nthread": 1,
            "seed": 264,
        },
        xgb.DMatrix(training, label=[0.2, -0.1], feature_names=list(names)),
        num_boost_round=2,
    )
    model_path = bundle / "model.ubj"
    booster.save_model(model_path)
    feature_path = bundle / "feature_contract.json"
    _write_json(feature_path, _feature_contract())
    cost_path = bundle / "cost_contract.json"
    _write_json(cost_path, _cost_contract())
    eligibility_path = bundle / "calibration_eligibility.json"
    _write_json(
        eligibility_path,
        {
            "schema_version": CALIBRATION_ELIGIBILITY_SCHEMA_VERSION,
            "candidate_id": "synthetic-passing-candidate",
            "lineage_id": "BTC-15M-cost-aware-market-residual-v99",
            "score_semantics": "direct_after_cost_action_value",
            "calibration_method": "identity_no_post_hoc_calibration",
            "calibration_fit_population": (
                "strictly_prior_market_grouped_rolling_origin_oof"
            ),
            "fixed_acceptance_threshold": 0.0,
            "threshold_or_parameter_search_performed": False,
            "route_side_missingness_or_outlier_filtering_performed": False,
            "all_frozen_oof_gates_passed": True,
            "offline_runtime_eligible": True,
            "live_shadow_authorized": False,
            "paper_or_live_execution_authorized": False,
            "safety": SAFETY,
        },
    )
    freeze_path = bundle / "candidate_freeze.json"
    _write_json(
        freeze_path,
        {
            "candidate_id": "synthetic-passing-candidate",
            "lineage_id": "BTC-15M-cost-aware-market-residual-v99",
            "all_gates_passed": True,
            "candidate_freeze_allowed": True,
            "promotion_evidence_eligible": False,
            "live_shadow_start_allowed": False,
            "safety": SAFETY,
        },
    )
    manifest_path = bundle / "runtime_bundle_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "candidate_id": "synthetic-passing-candidate",
            "lineage_id": "BTC-15M-cost-aware-market-residual-v99",
            "market_contract": {
                "family": "btc_updown_15m",
                "horizon_seconds": 900,
                "sides": ["UP", "DOWN"],
            },
            "model": {
                "path": model_path.relative_to(tmp_path).as_posix(),
                "sha256": sha256_file(model_path),
                "format": "xgboost_ubj",
                "objective": "reg:squarederror",
                "xgboost_version": xgb.__version__,
                "ordered_feature_names_sha256": canonical_json_sha256(
                    list(FEATURE_NAMES)
                ),
            },
            "feature_contract": _descriptor(feature_path, tmp_path),
            "cost_contract": _descriptor(cost_path, tmp_path),
            "calibration_eligibility": _descriptor(eligibility_path, tmp_path),
            "candidate_freeze": _descriptor(freeze_path, tmp_path),
            "decision_contract": {
                "actions": list(ACTIONS),
                "NO_TRADE_value": 0.0,
                "accept_if": "highest_side_score>0",
                "fixed_acceptance_threshold": 0.0,
                "side_tie_break_order": ["UP", "DOWN"],
                "one_trade_maximum_per_market": True,
            },
            "freshness_contract": {
                "maximum_decision_lag_ms": 5_000,
                "maximum_source_age_ms": 5_000,
                "stale_input_action": "NO_TRADE",
            },
            "runtime_authorization": {
                "offline_readiness_only": True,
                "live_shadow_authorized": False,
                "paper_or_live_execution_authorized": False,
                "wallet_signing_authorized": False,
                "polymarket_write_authorized": False,
                "capital_at_risk": False,
            },
            "safety": SAFETY,
        },
    )
    return {
        "manifest_path": manifest_path,
        "manifest_sha": sha256_file(manifest_path),
        "model_path": model_path,
        "feature_path": feature_path,
        "cost_path": cost_path,
        "eligibility_path": eligibility_path,
        "freeze_path": freeze_path,
        "booster": booster,
    }


def _feature_contract() -> dict:
    return {
        "base_feature_names": list(BASE_FEATURE_NAMES),
        "ordered_model_feature_contract": {
            "base_feature_count": len(BASE_FEATURE_NAMES),
            "missing_indicator_count": len(BASE_FEATURE_NAMES),
            "ordered_feature_count": len(FEATURE_NAMES),
            "ordered_feature_names_sha256": canonical_json_sha256(list(FEATURE_NAMES)),
        },
        "causality": {
            "market_horizon_seconds": 900,
            "available_at_ts_must_be_lte_decision_ts": True,
            "feature_cutoff_ts_must_be_lte_decision_ts": True,
            "max_input_ts_must_be_lte_decision_ts": True,
            "settlement_outcome_allowed": False,
            "target_or_pnl_allowed": False,
        },
        "missingness": {
            "explicit_indicator_for_every_base_feature": True,
            "missing_encoded_as_numeric_zero_allowed": False,
            "native_model_missing_value": "nan",
        },
    }


def _cost_contract() -> dict:
    return {
        "execution_policy": "HOLD_TO_SETTLEMENT",
        "action_policy": {
            "fixed_acceptance_threshold": 0.0,
            "threshold_search_allowed": False,
            "one_trade_maximum_per_market": True,
            "side_filter_allowed": False,
        },
        "cost_semantics": {
            "complement_quote_proxy_allowed": False,
            "true_paired_executable_asks_required": True,
            "unit_net_pnl": "gross_price_edge_minus_total_cost",
        },
        "sizing": {
            "unit_sizing": True,
            "sizing_multiplier": 1.0,
            "dynamic_sizing_allowed": False,
        },
        "safety": SAFETY,
    }


def _feature_row() -> dict:
    raw: dict[str, float | None] = {}
    for side_index, side in enumerate(("up", "down"), start=1):
        for suffix_index, suffix in enumerate(SIDE_RAW_SUFFIXES, start=1):
            raw[_side_raw_name(side, suffix)] = float(side_index + suffix_index / 100)
    raw.update(
        {
            "up_ask": 0.52,
            "up_bid": 0.50,
            "up_mid": 0.51,
            "down_ask": 0.50,
            "down_bid": 0.48,
            "down_mid": 0.49,
            "up_down_ask_sum": 1.02,
            "up_down_bid_sum": 0.98,
            "up_down_mid_sum": 1.0,
            "combined_spread_bps": 400.0,
            "chainlink_reference_distance_at_decision": 0.001,
            "btc_mid_price": 60_010.0,
            "chainlink_price_at_decision": 60_000.0,
            "btc_return_10s": 0.0001,
            "btc_return_30s": 0.0002,
            "btc_return_1m": 0.0003,
            "btc_return_5m": 0.001,
            "btc_return_15m": 0.002,
            "btc_volatility_1m": 0.001,
            "btc_volatility_5m": 0.002,
            "btc_volatility_15m": 0.003,
            "market_age_seconds": 300.0,
            "time_to_close_seconds": 600.0,
            "horizon_ms": 900_000.0,
            "provider_health_score": 1.0,
            "book_snapshot_pair_ts_delta_ms": 10.0,
        }
    )
    dependencies = set(raw)
    for values in GLOBAL_RAW_DEPENDENCIES.values():
        dependencies.update(values)
    provenance = {
        name: {"available_at_ts": 999_500, "max_input_ts": 999_500}
        for name in dependencies
    }
    return {
        "market_id": "synthetic-btc-15m",
        "market_family": "btc_updown_15m",
        "horizon_ms": 900_000,
        "market_start_ts": 700_000,
        "market_end_ts": 1_600_000,
        "decision_ts": 1_000_000,
        "available_at_ts": 999_500,
        "feature_cutoff_ts": 999_500,
        "max_input_ts": 999_500,
        "features": raw,
        "feature_provenance": provenance,
    }


def _causal_stream_fixture() -> tuple:
    start = 10_000_000
    decision = start + 300_000
    end = start + 900_000
    market = PolymarketCorpusMarket(
        market_id="stream-btc-15m",
        condition_id="condition-stream-btc-15m",
        slug="btc-updown-15m-stream",
        market_family="btc_updown_15m",
        horizon_ms=900_000,
        market_start_ts=start,
        market_end_ts=end,
        settlement_ts=end,
        up_token_id="up-token",
        down_token_id="down-token",
        reference_price_source="polymarket_rtds_chainlink",
        settlement_rule="btc_usd_at_close_gte_start",
        raw_market_sha256="a" * 64,
        trade_collection_mode="outcome_blind_stream",
        trade_stream_started_at_ts=start,
        trade_stream_ended_at_ts=decision,
        trade_stream_continuity_passed=True,
        trade_stream_timestamp_causality_violation_count=0,
        trade_api_request_failed=False,
        trade_rest_rows_truncated=False,
        trade_full_round_coverage_complete=True,
        trade_tape_censored=False,
    )
    books = []
    for outcome, token, bid, ask in (
        ("UP", "up-token", 0.50, 0.52),
        ("DOWN", "down-token", 0.48, 0.50),
    ):
        books.append(
            PolymarketCorpusBookSnapshot(
                market_id=market.market_id,
                token_id=token,
                outcome=outcome,
                ts=decision - 500,
                available_at_ts=decision - 400,
                bid_price=bid,
                ask_price=ask,
                mid_price=(bid + ask) / 2.0,
                bid_size=20.0,
                ask_size=20.0,
                liquidity_depth=1_000.0,
            )
        )
    for outcome, token in (("UP", "up-token"), ("DOWN", "down-token")):
        books.append(
            PolymarketCorpusBookSnapshot(
                market_id=market.market_id,
                token_id=token,
                outcome=outcome,
                ts=decision + 1,
                available_at_ts=decision + 1,
                bid_price=0.01,
                ask_price=0.99,
                mid_price=0.50,
                bid_size=1.0,
                ask_size=1.0,
                liquidity_depth=1.0,
            )
        )
    candles = []
    candle_ts = start - 900_000
    index = 0
    while candle_ts < decision:
        price = 60_000.0 + index
        candles.append(
            BinanceBTCCandle(
                ts=candle_ts,
                available_at_ts=candle_ts + 60_000,
                open_price=price,
                high_price=price + 1.0,
                low_price=price - 1.0,
                close_price=price + 0.5,
                volume=1.0,
                timeframe_ms=60_000,
            )
        )
        candle_ts += 60_000
        index += 1
    candles.append(
        BinanceBTCCandle(
            ts=decision,
            available_at_ts=decision + 60_000,
            open_price=1_000_000.0,
            high_price=1_000_001.0,
            low_price=999_999.0,
            close_price=1_000_000.5,
            volume=1.0,
            timeframe_ms=60_000,
        )
    )
    chainlink = [
        PolymarketChainlinkPrice(
            source_ts=start,
            available_at_ts=start,
            price=60_000.0,
        ),
        PolymarketChainlinkPrice(
            source_ts=decision - 1,
            available_at_ts=decision - 1,
            price=60_020.0,
        ),
        PolymarketChainlinkPrice(
            source_ts=decision + 1,
            available_at_ts=decision + 1,
            price=1_000_000.0,
        ),
    ]
    return market, tuple(books), tuple(candles), tuple(chainlink), decision


def _descriptor(path: Path, root: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
