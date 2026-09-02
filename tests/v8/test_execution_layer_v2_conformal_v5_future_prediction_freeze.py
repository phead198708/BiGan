from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_conformal_v5_future_evaluation import (
    FORBIDDEN_TARGET_FIELDS,
    _find_nonempty_fields,
)
from bigan.v8.polymarket.training.execution_layer_v2_conformal_v5_future_prediction_freeze import (
    ALLOWED_RAW_FEATURE_FILES,
    _materialize_future_action_rows,
    _materialize_selected_window_features,
)


def test_target_free_raw_window_builds_causal_complete_five_action_grid(
    tmp_path: Path,
) -> None:
    selected = _selected_row_fixture(tmp_path)
    features, opened = _materialize_selected_window_features([selected])
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2] / "examples/v8/polymarket_configs/"
            "execution_layer_v2_pairwise_action_advantage_lcb_feature_contract_v1.json"
        ).read_text(encoding="utf-8")
    )
    actions = _materialize_future_action_rows(
        features,
        selected_rows=[selected],
        feature_columns=tuple(contract["feature_columns"]),
    )

    assert len(features) == 5
    assert len(actions) == 25
    assert all(int(row["max_input_ts"]) <= int(row["decision_ts"]) for row in features)
    assert not _find_nonempty_fields(features, FORBIDDEN_TARGET_FIELDS)
    assert {row["action"] for row in actions} == {
        "BUY_UP_HOLD_TO_SETTLEMENT",
        "BUY_DOWN_HOLD_TO_SETTLEMENT",
        "BUY_UP_SELL_BEFORE_CLOSE",
        "BUY_DOWN_SELL_BEFORE_CLOSE",
        "NO_TRADE",
    }
    assert {row["role"] for row in actions} == {"future_unseen_evaluation"}
    assert opened[0]["resolution_artifact_opened"] is False
    assert set(opened[0]["raw_feature_artifacts"]) == set(ALLOWED_RAW_FEATURE_FILES)


def test_outcome_field_in_raw_feature_artifact_fails_closed(tmp_path: Path) -> None:
    selected = _selected_row_fixture(tmp_path)
    descriptor = selected["raw_artifacts"]["raw_polymarket_markets.jsonl"]
    path = Path(descriptor["path"])
    row = json.loads(path.read_text(encoding="utf-8"))
    row["settlement_outcome"] = "UP"
    _write_jsonl(path, [row])
    descriptor["sha256"] = _sha256(path)

    with pytest.raises(ValueError, match="forbidden targets"):
        _materialize_selected_window_features([selected])


def test_resolution_artifact_is_not_a_prediction_input(tmp_path: Path) -> None:
    selected = _selected_row_fixture(tmp_path)
    resolution_path = tmp_path / "raw_polymarket_resolutions.jsonl"
    _write_jsonl(resolution_path, [{"winning_outcome": "UP"}])
    selected["raw_artifacts"]["raw_polymarket_resolutions.jsonl"] = {
        "path": str(resolution_path),
        "sha256": _sha256(resolution_path),
        "row_count": 1,
    }

    features, opened = _materialize_selected_window_features([selected])

    assert features
    assert opened[0]["resolution_artifact_opened"] is False
    assert "raw_polymarket_resolutions.jsonl" not in opened[0]["raw_feature_artifacts"]


def _selected_row_fixture(tmp_path: Path) -> dict:
    start = 2_000_000
    end = start + 300_000
    market_id = "future-market-001"
    payloads: dict[str, list[dict]] = {
        "raw_polymarket_markets.jsonl": [
            {
                "market_id": market_id,
                "condition_id": "condition-001",
                "slug": "btc-updown-5m-future-001",
                "market_family": "btc_updown_5m",
                "horizon_ms": 300_000,
                "market_start_ts": start,
                "market_end_ts": end,
                "settlement_ts": end,
                "up_token_id": "up-token",
                "down_token_id": "down-token",
                "reference_price_source": "polymarket_rtds_chainlink",
                "settlement_rule": "UP if end reference is at least start reference",
                "paper_only": True,
                "capital_at_risk": False,
                "polymarket_write_enabled": False,
                "wallet_signing_enabled": False,
            }
        ],
        "raw_polymarket_orderbooks.jsonl": [],
        "raw_polymarket_trades.jsonl": [],
        "raw_binance_btcusdt_klines.jsonl": [],
        "raw_polymarket_chainlink_prices.jsonl": [],
    }
    for offset in range(-15, 5):
        ts = start + offset * 60_000
        payloads["raw_binance_btcusdt_klines.jsonl"].append(
            {
                "ts": ts,
                "available_at_ts": ts + 60_000,
                "open_price": 100_000.0 + offset,
                "high_price": 100_010.0 + offset,
                "low_price": 99_990.0 + offset,
                "close_price": 100_001.0 + offset,
                "volume": 1.0,
                "timeframe_ms": 60_000,
                "source": "binance_btcusdt",
            }
        )
    for offset in range(5):
        ts = start + offset * 60_000
        for outcome, bid, ask in (("UP", 0.54, 0.56), ("DOWN", 0.44, 0.46)):
            payloads["raw_polymarket_orderbooks.jsonl"].append(
                {
                    "market_id": market_id,
                    "token_id": f"{outcome.lower()}-token",
                    "outcome": outcome,
                    "ts": ts,
                    "available_at_ts": ts,
                    "bid_price": bid,
                    "ask_price": ask,
                    "mid_price": (bid + ask) / 2.0,
                    "bid_size": 100.0,
                    "ask_size": 100.0,
                    "liquidity_depth": 200.0,
                    "paper_only": True,
                    "capital_at_risk": False,
                    "polymarket_write_enabled": False,
                    "wallet_signing_enabled": False,
                }
            )
        payloads["raw_polymarket_chainlink_prices.jsonl"].append(
            {
                "source_ts": ts,
                "available_at_ts": ts,
                "price": 100_000.0 + offset,
                "source_type": "polymarket_rtds_chainlink",
                "symbol": "BTC/USD",
                "read_only": True,
                "paper_only": True,
                "capital_at_risk": False,
                "polymarket_write_enabled": False,
                "wallet_signing_enabled": False,
            }
        )
    descriptors = {}
    for filename, rows in payloads.items():
        path = tmp_path / filename
        _write_jsonl(path, rows)
        descriptors[filename] = {
            "path": str(path),
            "sha256": _sha256(path),
            "row_count": len(rows),
        }
    return {
        "scheduled_round_start_ts": start,
        "market_id": market_id,
        "entry_sha256": "e" * 64,
        "raw_artifacts": descriptors,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
