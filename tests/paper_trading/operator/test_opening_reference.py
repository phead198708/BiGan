from __future__ import annotations

import json
from dataclasses import asdict, replace

import pytest

from bigan.paper_trading.operator.config import OperatorConfig
from bigan.paper_trading.operator.discovery import MarketDiscoveryError, parse_gamma_markets
from bigan.paper_trading.operator.opening_reference import (
    OPENING_REFERENCE_ENDPOINT,
    bind_opening_reference,
    opening_reference_params,
)
from bigan.paper_trading.operator.runtime import PaperTradingOperator
from bigan.paper_trading.operator.transports import ChainlinkReadonlyFeed, GammaDiscoveryClient
from tests.paper_trading.operator.test_discovery import ENDPOINT, _filters, _row
from tests.paper_trading.operator.test_runtime import (
    FakeClock,
    FakeDiscovery,
    FakeResolution,
    _config,
    _emit_market_pair,
    _ready_operator,
    _selection,
)

START = 1_800_000


def twap_row(*, symbol="BTC", lookback=60, duration=900_000, start=START):
    row = _row("twap-market", start=start)
    row.update(underlying=symbol, slug=f"{symbol.lower()}-updown-{duration // 60000}m-{start // 1000}",
               windowDurationMs=duration, end_ts_ms=start + duration,
               resolutionSource=f"https://data.chain.link/streams/{symbol.lower()}-usd-twap-{lookback}s-streams",
               cryptoMarketConfig={"asset": symbol.lower(), "duration": f"{duration // 60000}m",
                                   "twapEnabled": True, "twapLookbackSeconds": lookback})
    return row


def twap_market(**kwargs):
    return parse_gamma_markets([twap_row(**kwargs)], source_endpoint=ENDPOINT, discovered_at_ms=START + 1000)[0]


def response(**changes):
    return {"openPrice": 100.0, "closePrice": None, "timestamp": START + 1000,
            "completed": False, "incomplete": True, "cached": True, **changes}


def bound_market():
    return bind_opening_reference(twap_market(), response(), requested_at_ms=START + 1000,
                                  received_at_ms=START + 1001)


@pytest.mark.parametrize("symbol,duration,lookback", [("BTC", 900000, 60), ("ETH", 300000, 30)])
def test_opening_request_is_exactly_asset_window_and_twap_bound(symbol, duration, lookback):
    market = twap_market(symbol=symbol, duration=duration, lookback=lookback)
    params = opening_reference_params(market)
    assert params == {"symbol": symbol, "eventStartTime": "1970-01-01T00:30:00Z",
                      "endDate": "1970-01-01T00:45:00Z" if duration == 900000 else "1970-01-01T00:35:00Z",
                      "variant": "fifteen" if duration == 900000 else "fiveminute",
                      "twapEnabled": "true", "twapLookbackSeconds": str(lookback)}
    bound = bind_opening_reference(market, response(closePrice=99999), requested_at_ms=START + 1000,
                                   received_at_ms=START + 1001)
    assert bound.reference_price_at_start == 100  # Never closePrice.
    assert bound.opening_reference.lookback_seconds == lookback
    assert bound.opening_reference.source_ts_ms == START + 1000
    assert len(bound.opening_reference.payload_sha256) == 64
    assert type(bound)(**json.loads(json.dumps(asdict(bound)))) == bound


@pytest.mark.parametrize("changes", [
    {"openPrice": None}, {"openPrice": 0}, {"openPrice": -1}, {"openPrice": True},
    {"openPrice": float("nan")}, {"openPrice": float("inf")},
    {"openPrice": "100"}, {"completed": True}, {"cached": 1},
    {"timestamp": START - 1}, {"timestamp": START + 1002}, {"timestamp": True},
    {"other": "unrecognized-schema"},
])
def test_bad_or_unavailable_opening_response_is_not_a_price(changes):
    with pytest.raises(ValueError):
        bind_opening_reference(twap_market(), response(**changes), requested_at_ms=START + 1000,
                               received_at_ms=START + 1001)


@pytest.mark.parametrize("requested,received", [(START - 1, START + 1001),
                                              (START + 1000, START + 900000),
                                              (START + 1001, START + 1000)])
def test_reference_request_cannot_precede_window_or_finish_after_expiry(requested, received):
    with pytest.raises(ValueError):
        bind_opening_reference(twap_market(), response(), requested_at_ms=requested, received_at_ms=received)


def test_conflicting_gamma_reference_and_changed_proof_are_rejected():
    with pytest.raises(ValueError, match="conflict"):
        bind_opening_reference(replace(twap_market(), reference_price_at_start=101), response(),
                               requested_at_ms=START + 1000, received_at_ms=START + 1001)
    bound = bound_market()
    for changes in ({"price": 101}, {"symbol": "ETH"}, {"condition_id": "other"}):
        with pytest.raises(ValueError, match="disagrees"):
            replace(bound, opening_reference=replace(bound.opening_reference, **changes))


@pytest.mark.parametrize("changes", [
    {"asset": "eth"}, {"duration": "5m"}, {"twapLookbackSeconds": True},
    {"twapLookbackSeconds": 300}, {"twapLookbackSeconds": 30}, {"twapEnabled": False},
])
def test_market_twap_metadata_must_agree_with_resolution_source(changes):
    row = twap_row()
    row["cryptoMarketConfig"].update(changes)
    with pytest.raises(MarketDiscoveryError):
        parse_gamma_markets([row], source_endpoint=ENDPOINT, discovered_at_ms=START + 1000)


async def test_discovery_fetches_current_reference_but_never_future_opening(monkeypatch):
    calls = []

    class HTTP:
        async def get_json(self, endpoint, *, params):
            calls.append((endpoint, params))
            if endpoint == OPENING_REFERENCE_ENDPOINT:
                return response()
            if params["slug"].endswith("-1800"):
                return [twap_row()]
            return []

    monkeypatch.setattr("bigan.paper_trading.operator.opening_reference.time.time_ns", lambda: (START + 1001) * 1000000)
    selected = await GammaDiscoveryClient(endpoint=ENDPOINT, http=HTTP()).discover(
        filters=_filters(), now_ms=START + 1000,
    )
    assert selected.current.opening_reference is not None
    assert sum(endpoint == OPENING_REFERENCE_ENDPOINT for endpoint, _ in calls) == 1
    calls.clear()
    selected = await GammaDiscoveryClient(endpoint=ENDPOINT, http=HTTP()).discover(
        filters=_filters(), now_ms=START - 1000,
    )
    assert selected.current is None and selected.next.opening_reference is None
    assert all(endpoint != OPENING_REFERENCE_ENDPOINT for endpoint, _ in calls)


async def test_operator_persists_reference_and_resumes_without_refetch(tmp_path):
    market = bound_market()
    config, clock = _config(tmp_path), FakeClock(START + 1001)
    operator = PaperTradingOperator(config=config, discovery=FakeDiscovery([_selection(market)]),
                                    resolution=FakeResolution([None]), clock_ms=clock)
    await operator.start()
    assert operator.session is not None
    assert operator.session.runner.pricing_engine.reference_model == "published_twap"
    checkpoint = json.loads(operator.checkpoint_store.path.read_text())
    assert checkpoint["market"]["opening_reference"]["payload_sha256"] == market.opening_reference.payload_sha256
    assert operator._session_config(market)["market_identity"]["opening_reference"] == asdict(market.opening_reference)
    run_id = operator.run_id
    await operator.shutdown()
    discovery = FakeDiscovery([])  # Any attempt to rediscover would fail.
    restored = PaperTradingOperator(config=config, discovery=discovery, resolution=FakeResolution([None]), clock_ms=clock)
    try:
        await restored.start()
        assert restored.run_id == run_id
        assert restored.active_market.opening_reference == market.opening_reference
        assert discovery.calls == 0
    finally:
        await restored.shutdown()


def test_opening_endpoint_has_strict_allowlist(tmp_path):
    with pytest.raises(ValueError):
        OperatorConfig(operator_id="test", strategy_id="test", paper_account_id="test", source_commit="test",
                       output_dir=tmp_path, opening_reference_endpoint="https://example.com/price")


async def test_twap_operator_reaches_running_only_after_matching_twap_samples(tmp_path):
    market, clock = bound_market(), FakeClock(START + 1001)
    operator = PaperTradingOperator(config=_config(tmp_path), discovery=FakeDiscovery([_selection(market)]),
                                    resolution=FakeResolution([None]), clock_ms=clock)
    try:
        await operator.start()
        await _ready_operator(operator, clock)  # Legacy spot-oracle sample must be rejected.
        assert not operator.status().pricing_inputs["ready"]
        assert operator.session.runner.oms_calls == 0
        feed = ChainlinkReadonlyFeed(
            expected_symbol="btc/usd", source=operator._oracle_source, lookback_seconds=60,
            on_sample=lambda sample, generation: operator.ingest_oracle(sample, generation=generation),
        )
        for ts, price in [(START + 1000, 100), (START + 1001, 101)]:
            await feed.on_raw(json.dumps({"topic": "crypto_prices_twap_sixty", "type": "update", "timestamp": ts,
                "payload": {"symbol": "btc/usd", "timestamp": ts, "window_s": 60,
                            "full_accuracy_value": str(price * 10**18)}}),
                generation=operator.generation, received_at_ms=clock.now_ms)
        assert operator.status().state.value == "RUNNING"
        clock.now_ms += 1
        await _emit_market_pair(operator, market, timestamp_ms=clock.now_ms, sequence=2)
        assert operator.session.runner.execution_history[-1].status == "FILLED"
        decision = operator.session.runner.last_decision
        assert decision.oracle_twap_so_far == 101
        assert decision.twap_weight == 0
        assert decision.effective_strike == 100
        assert decision.spot_price == 102  # Binance midpoint is still recorded separately.
        assert operator.session.current_snapshot.cash < operator.config.initial_bankroll
    finally:
        await operator.shutdown()
