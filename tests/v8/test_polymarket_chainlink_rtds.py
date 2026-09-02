from __future__ import annotations

import asyncio
import json

import pytest

from bigan.v8.polymarket.recorder import chainlink_rtds as chainlink_rtds_module
from bigan.v8.polymarket.recorder.chainlink_rtds import (
    CHAINLINK_RTDS_RAW_ROW_SCHEMA_VERSION,
    ChainlinkRTDSMessageError,
    PolymarketChainlinkRTDSCollector,
    _price_stream_is_stale,
    parse_chainlink_rtds_message,
)
from bigan.v8.polymarket.training.execution_layer_v2_one_hour_goal import (
    _write_per_round_artifacts,
)
from bigan.v8.polymarket.training.o_v8_paper_fresh_loop import (
    _fresh_public_rows_from_provider_payloads,
    _provider_chainlink_regime_features,
)


def test_chainlink_rtds_parser_normalizes_snapshot_and_update_causally() -> None:
    snapshot = {
        "payload": {
            "data": [
                {"timestamp": 3_000_000, "value": 65_000.0},
                {"timestamp": 3_001_000, "value": 65_010.0},
            ],
            "symbol": "btc/usd",
        },
        "timestamp": 3_002_000,
        "topic": "crypto_prices",
        "type": "subscribe",
    }
    update = {
        "connection_id": "pytest",
        "payload": {
            "full_accuracy_value": "65020000000000000000000",
            "symbol": "btc/usd",
            "timestamp": 3_003_000,
            "value": 65_020.0,
        },
        "timestamp": 3_003_500,
        "topic": "crypto_prices_chainlink",
        "type": "update",
    }

    snapshot_rows = parse_chainlink_rtds_message(
        json.dumps(snapshot), received_at_ts=3_002_500
    )
    update_rows = parse_chainlink_rtds_message(
        json.dumps(update), received_at_ts=3_003_600
    )

    assert len(snapshot_rows) == 2
    assert len(update_rows) == 1
    assert snapshot_rows[0]["schema_version"] == CHAINLINK_RTDS_RAW_ROW_SCHEMA_VERSION
    assert snapshot_rows[0]["source_message_type"] == (
        "chainlink_subscription_snapshot"
    )
    assert update_rows[0]["source_message_type"] == "chainlink_update"
    assert update_rows[0]["available_at_ts"] == 3_003_600
    assert all(row["source_ts"] <= row["available_at_ts"] for row in snapshot_rows)
    assert all(row["timestamp_causality_valid"] is True for row in update_rows)
    assert all(row["paper_only"] is True for row in [*snapshot_rows, *update_rows])
    assert all(row["capital_at_risk"] is False for row in [*snapshot_rows, *update_rows])


def test_chainlink_rtds_parser_fails_closed_on_invalid_price() -> None:
    payload = {
        "payload": {
            "symbol": "btc/usd",
            "timestamp": 3_000_000,
            "value": 0.0,
        },
        "timestamp": 3_000_100,
        "topic": "crypto_prices_chainlink",
        "type": "update",
    }

    with pytest.raises(ChainlinkRTDSMessageError) as exc_info:
        parse_chainlink_rtds_message(
            json.dumps(payload), received_at_ts=3_000_200
        )

    assert exc_info.value.reason_code == (
        "chainlink_rtds_price_missing_or_non_positive"
    )


def test_chainlink_regime_features_exclude_post_decision_rows() -> None:
    rows = [
        _chainlink_row(source_ts=3_000_000, available_at_ts=3_001_000, price=65_000.0),
        _chainlink_row(source_ts=3_059_000, available_at_ts=3_060_000, price=65_130.0),
        _chainlink_row(source_ts=3_061_000, available_at_ts=3_061_000, price=1.0),
    ]

    features = _provider_chainlink_regime_features(
        rows=rows,
        market_start_ts=3_000_000,
        decision_ts=3_060_000,
        comparison_btc_price=65_120.0,
    )

    assert features["chainlink_price_at_decision"] == 65_130.0
    assert features["chainlink_reference_price_at_market_start"] == 65_000.0
    assert features["chainlink_reference_distance_at_decision"] == pytest.approx(
        0.002
    )
    provenance = features["chainlink_regime_feature_provenance"]
    assert provenance["provenance_valid"] is True
    assert provenance["max_input_ts"] <= 3_060_000


def test_fresh_provider_uses_chainlink_reference_proxy_with_provenance() -> None:
    market = {
        "market_id": "chainlink-market",
        "condition_id": "chainlink-condition",
        "slug": "btc-updown-5m-3000",
        "market_family": "btc_updown_5m",
        "market_start_ts": 3_000_000,
        "market_end_ts": 3_300_000,
        "horizon_ms": 300_000,
        "up_token_id": "up-token",
        "down_token_id": "down-token",
        "reference_price_start": None,
        "reference_price_at_start": None,
    }
    books = [
        _book_row("UP", "up-token", 0.58, 0.60),
        _book_row("DOWN", "down-token", 0.40, 0.42),
    ]
    candles = [
        {
            "ts": 3_000_000,
            "close_time": 3_060_000,
            "available_at_ts": 3_060_000,
            "open_price": 65_000.0,
            "close_price": 65_120.0,
        }
    ]
    chainlink_rows = [
        _chainlink_row(source_ts=3_000_000, available_at_ts=3_001_000, price=65_000.0),
        _chainlink_row(source_ts=3_059_000, available_at_ts=3_060_500, price=65_130.0),
    ]

    rows = _fresh_public_rows_from_provider_payloads(
        run_id="chainlink-feature",
        markets=[market],
        orderbooks=books,
        trades=[],
        btc_candles=candles,
        chainlink_rtds_prices=chainlink_rows,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["decision_ts"] == 3_060_500
    assert row["reference_price_to_beat_at_decision"] == 65_000.0
    assert row["reference_price_to_beat_distance_at_decision"] == pytest.approx(
        0.002
    )
    assert row["chainlink_price_at_decision"] == 65_130.0
    assert row["decision_time_regime_feature_max_input_ts"] <= row["decision_ts"]
    provenance = row["reference_price_to_beat_distance_provenance"]
    assert provenance["provenance_valid"] is True
    assert provenance["reference_price_to_beat_source_type"] == (
        "polymarket_rtds_chainlink_market_start_proxy"
    )


def test_round_artifacts_persist_chainlink_rows_and_hash(tmp_path) -> None:
    market_id = "chainlink-round"
    manifest = _write_per_round_artifacts(
        goal_dir=tmp_path,
        intents=[],
        fills=[],
        ledger_rows=[],
        settlement_rows=[],
        trace_rows=[
            {
                "market_id": market_id,
                "decision_ts": 3_060_000,
                "market_start_ts": 3_000_000,
            }
        ],
        raw_market_rows=[
            {"market_id": market_id, "market_start_ts": 3_000_000}
        ],
        raw_orderbook_rows=[],
        raw_trade_rows=[],
        raw_btc_candle_rows=[],
        raw_chainlink_price_rows=[
            _chainlink_row(
                source_ts=3_000_000,
                available_at_ts=3_001_000,
                price=65_000.0,
            )
        ],
    )

    round_row = manifest["round_artifact_rows"][0]
    assert round_row["raw_chainlink_price_row_count"] == 1
    assert "raw_polymarket_chainlink_prices" in round_row["artifact_paths"]
    assert "raw_polymarket_chainlink_prices" in round_row["artifact_hashes"]
    assert manifest["per_round_chainlink_covered_market_count"] == 1


def test_chainlink_collector_deduplicates_replayed_snapshot_rows() -> None:
    collector = PolymarketChainlinkRTDSCollector(max_rows=10)
    payload = json.dumps(
        {
            "payload": {
                "data": [{"timestamp": 3_000_000, "value": 65_000.0}],
                "symbol": "btc/usd",
            },
            "timestamp": 3_000_100,
            "topic": "crypto_prices",
            "type": "subscribe",
        }
    )

    collector._accept_message(payload)
    collector._accept_message(payload)

    assert len(collector.rows()) == 1
    assert collector.collection_report()["raw_price_row_count"] == 1
    assert collector.collection_report()["timestamp_causality_violation_count"] == 0


def test_chainlink_price_stream_staleness_boundary_is_deterministic() -> None:
    assert _price_stream_is_stale(
        now_monotonic=14.999,
        last_price_row_monotonic=0.0,
        stale_reconnect_seconds=15.0,
    ) is False
    assert _price_stream_is_stale(
        now_monotonic=15.0,
        last_price_row_monotonic=0.0,
        stale_reconnect_seconds=15.0,
    ) is True


def test_chainlink_collector_reconnects_silent_open_price_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = PolymarketChainlinkRTDSCollector(
        max_rows=10,
        receive_poll_seconds=0.001,
        ping_interval_seconds=0.001,
        reconnect_delay_seconds=0.001,
        stale_reconnect_seconds=0.005,
    )
    payload = json.dumps(
        {
            "payload": {
                "symbol": "btc/usd",
                "timestamp": 3_003_000,
                "value": 65_020.0,
            },
            "timestamp": 3_003_500,
            "topic": "crypto_prices_chainlink",
            "type": "update",
        }
    )
    connection_index = 0

    class FakeSocket:
        def __init__(self, index: int) -> None:
            self.index = index

        async def send(self, _payload: str) -> None:
            return None

        async def recv(self) -> str:
            if self.index == 1:
                await asyncio.sleep(0.002)
                raise TimeoutError
            collector._stop_event.set()
            return payload

    class FakeConnection:
        def __init__(self, index: int) -> None:
            self.socket = FakeSocket(index)

        async def __aenter__(self) -> FakeSocket:
            return self.socket

        async def __aexit__(self, *_args: object) -> None:
            return None

    def fake_connect(*_args: object, **_kwargs: object) -> FakeConnection:
        nonlocal connection_index
        connection_index += 1
        return FakeConnection(connection_index)

    monkeypatch.setattr(chainlink_rtds_module.websockets, "connect", fake_connect)

    asyncio.run(collector._run())

    report = collector.collection_report()
    assert report["connection_count"] == 2
    assert report["reconnect_count"] == 1
    assert report["stale_reconnect_count"] == 1
    assert report["stale_reconnect_seconds"] == 0.005
    assert report["raw_price_row_count"] == 1
    assert report["last_price_row_source_ts"] == 3_003_000
    assert report["last_error_type"] == "ChainlinkRTDSStaleStreamError"


def _chainlink_row(
    *,
    source_ts: int,
    available_at_ts: int,
    price: float,
) -> dict[str, object]:
    return {
        "source_type": "polymarket_rtds_chainlink",
        "source_ts": source_ts,
        "available_at_ts": available_at_ts,
        "price": price,
        "paper_only": True,
        "capital_at_risk": False,
    }


def _book_row(
    outcome: str,
    token_id: str,
    bid: float,
    ask: float,
) -> dict[str, object]:
    return {
        "market_id": "chainlink-market",
        "token_id": token_id,
        "outcome": outcome,
        "ts": 3_060_000,
        "available_at_ts": 3_060_000,
        "bid_price": bid,
        "ask_price": ask,
        "mid_price": (bid + ask) / 2.0,
        "bid_size": 2.0,
        "ask_size": 2.0,
        "liquidity_depth": 4.0,
    }
