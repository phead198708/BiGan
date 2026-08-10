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
from bigan.v8.polymarket.cost_aware_residual_v4_stacking import (
    DEFAULT_OUTPUT_DIR as V4_CHALLENGER_OUTPUT_DIR,
)
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.pooled_residual_runtime import (
    ACTIONS,
    BUNDLE_SCHEMA_VERSION,
    PooledResidualRuntimeError,
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
        row["market_end_ts"] = row["market_start_ts"] + 300_000
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
        "market_start_ts": 700_000,
        "market_end_ts": 1_600_000,
        "decision_ts": 1_000_000,
        "available_at_ts": 999_500,
        "feature_cutoff_ts": 999_500,
        "max_input_ts": 999_500,
        "features": raw,
        "feature_provenance": provenance,
    }


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
