from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pytest

from bigan.v8.polymarket.residual_promotion_live_signal_preview import (
    LiveSignalPreviewError,
    _market_row_after_trade_collection,
    signal_from_feature_row,
    write_live_preview,
)


class _Runtime:
    maximum_source_age_ms = 5_000

    def score_feature_row(self, row, *, observed_at_ts):
        assert observed_at_ts == row["decision_ts"]
        return {
            "selected_action": "BUY_UP_HOLD",
            "action_values": {
                "BUY_UP_HOLD": np.float64(0.125),
                "BUY_DOWN_HOLD": np.float64(-0.1455),
                "NO_TRADE": np.float64(0.0),
            },
        }


def _market() -> dict:
    start = 1_800_000_000_000
    return {
        "market_id": "market-live-preview",
        "slug": f"btc-updown-15m-{start // 1000}",
        "market_family": "btc_updown_15m",
        "market_start_ts": start,
        "market_end_ts": start + 900_000,
    }


def test_live_preview_publishes_first_signal_before_round_close(tmp_path) -> None:
    market = _market()
    signal = signal_from_feature_row(
        feature_row={
            "market_id": market["market_id"],
            "decision_ts": market["market_start_ts"] + 300_000,
        },
        runtime=_Runtime(),
        decision_number=1,
        already_accepted=False,
    )
    output = tmp_path / "outcome_blind_live_signal_preview.json"
    report = write_live_preview(
        output_path=output,
        candidate_bundle_sha256="a" * 64,
        market=market,
        signals=[signal],
        generated_at=datetime.fromtimestamp(
            (market["market_start_ts"] + 301_000) / 1000,
            tz=UTC,
        ),
    )

    assert report["round_state"] == "in_progress"
    assert report["signals"] == [signal]
    assert report["accepted_action"] == "BUY_UP_HOLD"
    assert report["preview_is_provisional"] is True
    assert report["monitoring_influences_collection"] is False
    assert report["fresh_outcomes_accessed"] is False
    assert report["outcomes_accessed"] is False
    assert report["settlement_accessed"] is False
    assert report["pnl_accessed"] is False
    assert report["wallet_signing_allowed"] is False
    assert report["polymarket_write_allowed"] is False
    assert report["capital_at_risk"] is False
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_live_preview_requires_sequential_frozen_decision_times(tmp_path) -> None:
    market = _market()
    with pytest.raises(LiveSignalPreviewError, match="signal contract"):
        write_live_preview(
            output_path=tmp_path / "preview.json",
            candidate_bundle_sha256="a" * 64,
            market=market,
            signals=[
                {
                    "decision_number": 2,
                    "decision_ts": market["market_start_ts"] + 600_000,
                    "selected_action": "NO_TRADE",
                    "accepted_at_this_decision": False,
                    "action_values": {
                        "BUY_UP_HOLD": -0.1,
                        "BUY_DOWN_HOLD": -0.2,
                        "NO_TRADE": 0.0,
                    },
                }
            ],
        )


def test_live_preview_never_accepts_twice() -> None:
    market = _market()
    second = signal_from_feature_row(
        feature_row={
            "market_id": market["market_id"],
            "decision_ts": market["market_start_ts"] + 600_000,
        },
        runtime=_Runtime(),
        decision_number=2,
        already_accepted=True,
    )
    assert second["selected_action"] == "BUY_UP_HOLD"
    assert second["accepted_at_this_decision"] is False


class _FailClosedRuntime:
    def score_feature_row(self, row, *, observed_at_ts):
        return {
            "selected_action": "NO_TRADE",
            "action_values": {
                "BUY_UP_HOLD": None,
                "BUY_DOWN_HOLD": None,
                "NO_TRADE": 0.0,
            },
            "fail_closed": True,
            "fail_closed_reasons": ["runtime source input is stale"],
        }


def test_live_preview_exposes_runtime_fail_closed_reason_without_fake_signal() -> None:
    market = _market()
    with pytest.raises(
        LiveSignalPreviewError,
        match="runtime source input is stale",
    ):
        signal_from_feature_row(
            feature_row={
                "market_id": market["market_id"],
                "decision_ts": market["market_start_ts"] + 300_000,
            },
            runtime=_FailClosedRuntime(),
            decision_number=1,
            already_accepted=False,
        )


def test_market_row_is_frozen_after_trade_provider_metadata_exists() -> None:
    market = {
        **_market(),
        "condition_id": "condition-live-preview",
        "horizon_ms": 900_000,
        "settlement_ts": _market()["market_end_ts"],
        "up_token_id": "up-token",
        "down_token_id": "down-token",
        "reference_price_source": "chainlink_btc_usd",
        "settlement_rule": "strictly_above_reference_is_up",
        "trade_collection_mode": "polymarket_data_api_paginated",
        "trade_api_collection_ts": _market()["market_start_ts"] + 299_000,
        "trade_api_request_failed": False,
        "trade_rest_rows_truncated": False,
    }
    raw = _market_row_after_trade_collection(market)
    assert raw["trade_collection_mode"] == "polymarket_data_api_paginated"
    assert raw["trade_api_collection_ts"] == market["trade_api_collection_ts"]

    del market["trade_collection_mode"]
    with pytest.raises(
        LiveSignalPreviewError,
        match="trade provider metadata",
    ):
        _market_row_after_trade_collection(market)
