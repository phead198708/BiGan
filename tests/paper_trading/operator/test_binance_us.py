"""Binance.US is an explicit, separately audited venue, never a host fallback."""

import asyncio
import hashlib
import json
from dataclasses import replace

import pytest

from bigan.features.binance_ofi import BinanceOFICalculator
from bigan.paper_trading.operator import live
from bigan.paper_trading.operator.config import (
    BINANCE_US_DEPTH_ENDPOINT,
    BINANCE_US_WS_URL,
    load_operator_config,
    operator_config_from_mapping,
)
from bigan.paper_trading.operator.live import LiveFeedSupervisor
from bigan.paper_trading.operator.pricing_inputs import ReferencePriceSample
from bigan.paper_trading.operator.runtime import PaperTradingOperator
from bigan.paper_trading.stack.preflight import Preflight
from bigan.paper_trading.stack.report import SoakReport
from tests.paper_trading.operator.test_config import _minimal
from tests.paper_trading.operator.test_runtime import (
    FakeClock,
    FakeDiscovery,
    FakeResolution,
    _config,
    _market,
    _ready_operator,
    _selection,
)
from tests.paper_trading.operator.test_transports import FakeHTTP

US = {"binance_venue": "us", "binance_depth_endpoint": BINANCE_US_DEPTH_ENDPOINT,
      "binance_ws_url": BINANCE_US_WS_URL}


def test_us_config_is_explicit_and_has_distinct_identity(tmp_path):
    default = operator_config_from_mapping(_minimal(tmp_path))
    us = operator_config_from_mapping(_minimal(tmp_path, **US))
    assert default.binance_venue == "global"
    assert default.binance_spot_source == "binance_depth:BTCUSDT"
    assert us.binance_spot_source == "binance_us_depth:BTCUSDT"
    assert default.config_sha256 != us.config_sha256
    assert us.binance_source_identity()["display_name"] == "Binance.US"
    assert load_operator_config("config/paper_operator.live.example.toml").binance_venue == "us"


@pytest.mark.parametrize("changes", [
    {"binance_venue": "global"}, {"binance_venue": "US"}, {"binance_venue": None},
    {"binance_venue": ["us"]},
    {"binance_depth_endpoint": "https://api.binance.com/api/v3/depth"},
    {"binance_ws_url": "wss://stream.binance.com:9443/ws"},
    {"binance_depth_endpoint": "https://api.binance.us/api/v3/order"},
    {"binance_depth_endpoint": "https://api.binance.us:8080/api/v3/depth"},
    {"binance_ws_url": "wss://stream.binance.us:8888/ws"},
    {"binance_ws_url": "wss://stream.binance.us.evil.example:9443/ws"},
    {"binance_ws_url": "wss://user:secret@stream.binance.us:9443/ws"},
    {"binance_depth_endpoint": "https://api.binance.us/api/v3/depth?symbol=ETHUSDT"},
])
def test_us_mixed_venue_or_unapproved_endpoints_fail_before_io(tmp_path, changes):
    output = tmp_path / "never-created"
    with pytest.raises(ValueError):
        operator_config_from_mapping(_minimal(output, **(US | changes)))
    assert not output.exists()


@pytest.mark.parametrize("changes", [
    {"binance_venue": "us"},
    {"binance_depth_endpoint": BINANCE_US_DEPTH_ENDPOINT},
    {"binance_ws_url": BINANCE_US_WS_URL},
])
def test_partial_change_of_global_configuration_is_rejected(tmp_path, changes):
    with pytest.raises(ValueError, match="match binance_venue"):
        operator_config_from_mapping(_minimal(tmp_path, **changes))


@pytest.mark.parametrize("symbol", ["BTCUSDT", "ETHUSDT"])
def test_ofi_identity_includes_venue_without_changing_math(symbol):
    global_ofi = BinanceOFICalculator(symbol=symbol, zscore_min_samples=1)
    us_ofi = BinanceOFICalculator(symbol=symbol, venue="us", zscore_min_samples=1)
    assert global_ofi.config_identity() != us_ofi.config_identity()
    assert us_ofi.config_identity()["venue"] == "us"
    for ts, qty in ((1, 2), (2, 4), (3, 3)):
        tick = {"bid_price": 100, "bid_qty": qty, "ask_price": 101, "ask_qty": 2, "ts_ms": ts}
        assert global_ofi.on_depth_update(**tick) == us_ofi.on_depth_update(**tick)
    with pytest.raises(ValueError, match="venue"):
        BinanceOFICalculator(venue="auto")


async def test_us_operator_binds_runner_manifest_pricing_status_and_report(tmp_path):
    config = replace(_config(tmp_path / "paper"), **US)
    clock, market = FakeClock(10_000), _market(1)
    operator = PaperTradingOperator(
        config=config, discovery=FakeDiscovery([_selection(market)]),
        resolution=FakeResolution([None]), clock_ms=clock,
    )
    await operator.start()
    try:
        assert operator.session.runner.ofi_engine.config_identity()["venue"] == "us"
        identity = {"strategy": operator.session.runner.config_identity(),
                    "session_config": operator._session_config(market)}
        expected_hash = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"),
                                                  allow_nan=False).encode()).hexdigest()
        assert operator.session.store.manifest.config_sha256 == expected_hash
        await _ready_operator(operator, clock)
        assert operator.pricing_provider.last_spot_sample.source == "binance_us_depth:BTCUSDT"
        assert not operator.pricing_provider.ingest_spot(
            ReferencePriceSample(10_001, 10_001, 102, "binance_depth:BTCUSDT")
        )
        status = operator.status()
        assert status.feeds["binance"]["venue"] == status.alpha["venue"] == "us"
        assert status.market_data["spot"]["venue"] == "us"
        assert status.market_data["spot"]["source"] == status.alpha["source"] == "binance_us_depth:BTCUSDT"
        check = Preflight(config, tmp_path / "config.toml", tmp_path, "127.0.0.1", 8088, None, True)
        report = SoakReport(check)
        assert report.data["market_data_source"] == check.summary()["market_data_source"]
        assert report.data["market_data_source"]["ws_endpoint"] == BINANCE_US_WS_URL
        assert "Market data: Binance.US" in report.markdown()
    finally:
        await operator.shutdown()
    # Never resume a Global manifest/checkpoint under the US source (or vice versa).
    global_config = replace(config, binance_venue="global",
                            binance_depth_endpoint="https://api.binance.com/api/v3/depth",
                            binance_ws_url="wss://stream.binance.com:9443/ws")
    other = PaperTradingOperator(config=global_config, discovery=FakeDiscovery([_selection(market)]),
                                 resolution=FakeResolution([None]), clock_ms=clock)
    try:
        await other.start()
        assert other.status().state.value == "FAILED"
        assert other.session is None
    finally:
        await other.shutdown()


async def test_live_wiring_uses_us_rest_and_ws_and_reboots_same_venue(tmp_path, monkeypatch):
    config = replace(_config(tmp_path), **US)
    operator = PaperTradingOperator(config=config, discovery=FakeDiscovery([_selection(_market(1))]),
                                    resolution=FakeResolution([None]), clock_ms=FakeClock(10_000))
    await operator.start()
    http = FakeHTTP({"lastUpdateId": 10, "bids": [["100", "2"]], "asks": [["101", "3"]]})
    captured = []

    class CaptureTransport:
        def __init__(self, **kwargs):
            captured.append(kwargs)

        async def run(self, _stop):
            return None

    monkeypatch.setattr(live, "PublicWebSocketTransport", CaptureTransport)
    tasks = LiveFeedSupervisor(operator=operator, http=http)._start_window_feeds(operator.generation, asyncio.Event())
    try:
        await asyncio.gather(*tasks)
        binance = captured[0]
        assert binance["endpoint"] == BINANCE_US_WS_URL
        assert binance["subscription"]["params"] == ["btcusdt@depth@100ms"]
        await binance["on_generation"](1)
        await binance["on_payload"]({"s": "BTCUSDT", "E": 9900, "U": 11, "u": 11,
                                     "b": [["100", "4"]], "a": []}, 1, 10_000)
        assert operator.pricing_provider.last_spot_sample.source == "binance_us_depth:BTCUSDT"
        assert operator.binance_sync.last_update_id == 11
        await binance["on_disconnect"]()
        await binance["on_generation"](2)
        assert http.calls == [(BINANCE_US_DEPTH_ENDPOINT, {"symbol": "BTCUSDT", "limit": 1000})] * 2
        assert operator.pricing_provider.last_spot_sample is None
    finally:
        await operator.shutdown()
