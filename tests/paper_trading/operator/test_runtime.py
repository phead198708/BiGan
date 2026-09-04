from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from bigan.paper_trading.operator.config import OperatorConfig
from bigan.paper_trading.operator.discovery import DiscoveredMarket, DiscoverySelection
from bigan.paper_trading.operator.ownership import AccountOwnershipError
from bigan.paper_trading.operator.pricing_inputs import ReferencePriceSample
from bigan.paper_trading.operator.read_model import (
    OperatorReadRepository,
    OperatorState,
    OperatorStatusWriter,
)
from bigan.paper_trading.operator.resolution import FinalResolution
from bigan.paper_trading.operator.runtime import PaperTradingOperator, stable_run_id
from bigan.paper_trading.session import PaperSessionFailedError, PaperTradingSession
from bigan.strategies.polymarket_pricing import PolymarketPricingEngine, SignalDirection


class FakeClock:
    def __init__(self, now_ms: int) -> None:
        self.now_ms = now_ms

    def __call__(self) -> int:
        return self.now_ms


class FakeDiscovery:
    def __init__(self, selections: list[DiscoverySelection]) -> None:
        self.selections = selections
        self.calls = 0

    async def discover(self, **_kwargs: object) -> DiscoverySelection:
        index = min(self.calls, len(self.selections) - 1)
        self.calls += 1
        return self.selections[index]


class FakeResolution:
    def __init__(self, values: list[FinalResolution | None]) -> None:
        self.values = values
        self.calls = 0

    async def resolve(self, *_args: object, **_kwargs: object) -> FinalResolution | None:
        index = min(self.calls, len(self.values) - 1)
        self.calls += 1
        return self.values[index]


class FailingStatusWriter(OperatorStatusWriter):
    def __init__(self, path: Path) -> None:
        super().__init__(path)

    def write(self, _status: object) -> None:
        raise OSError("projection disk full")


def _market(index: int, *, start: int = 0) -> DiscoveredMarket:
    end = start + 900_000
    return DiscoveredMarket(
        market_id=f"market-{index}",
        condition_id=f"condition-{index}",
        slug=f"btc-updown-15m-{index}",
        title=f"BTC up or down window {index}",
        underlying="BTC",
        market_type="binary_up_down",
        window_duration_ms=900_000,
        start_ts_ms=start,
        end_ts_ms=end,
        yes_token_id=f"yes-{index}",
        no_token_id=f"no-{index}",
        active=True,
        closed=False,
        accepting_orders=True,
        source_endpoint="https://gamma-api.polymarket.com/markets",
        discovered_at_ms=max(0, start),
        resolution_source="chainlink-btc-usd",
        resolution_identity=f"resolution-rule-{index}",
        reference_price_at_start=100.0,
        raw_payload_sha256=f"{index:064x}",
    )


def _selection(current: DiscoveredMarket, next_market: DiscoveredMarket | None = None):
    return DiscoverySelection(current=current, next=next_market, eligible_count=1 + (next_market is not None))


def _config(tmp_path: Path) -> OperatorConfig:
    return OperatorConfig(
        operator_id="operator-test",
        strategy_id="strategy-test",
        paper_account_id="paper-account-test",
        source_commit="deadbeef",
        output_dir=tmp_path,
        ofi_min_samples=1,
        volatility_min_samples=1,
        volatility_return_interval_ms=1,
        pricing_tail_cutoff_ms=1,
        pricing_min_edge_15m=0.0,
        max_alpha_age_ms=10_000,
        max_market_age_ms=10_000,
        max_pricing_age_ms=10_000,
        max_spread_allowed=0.10,
    )


def _final(market: DiscoveredMarket, *, received: int | None = None) -> FinalResolution:
    return FinalResolution(
        market_id=market.market_id,
        condition_id=market.condition_id,
        window_id=market.window_id,
        yes_payout=1.0,
        settlement_ts_ms=market.end_ts_ms,
        source="gamma:chainlink-btc-usd",
        source_ts_ms=market.end_ts_ms,
        received_ts_ms=market.end_ts_ms + 1 if received is None else received,
        source_reference=f"resolution:{market.market_id}",
        resolution_identity=market.resolution_identity,
    )


async def _ready_operator(
    operator: PaperTradingOperator,
    clock: FakeClock,
    *,
    produce_decision: bool = True,
) -> object | None:
    generation = operator.generation
    now = clock.now_ms
    await operator.ingest_binance_snapshot(
        {"lastUpdateId": 9, "bids": [["99", "2"]], "asks": [["101", "2"]]},
        generation=generation,
        received_at_ms=now - 1,
    )
    # REST seeds only the book; the first exchange event seeds pricing time.
    await operator.ingest_binance_delta(
        {"s": "BTCUSDT", "E": now - 1, "U": 10, "u": 10, "b": [], "a": []},
        generation=generation, received_at_ms=now,
    )
    await operator.ingest_binance_delta(
        {
            "s": "BTCUSDT",
            "E": now,
            "U": 11,
            "u": 11,
            "b": [["101", "4"]],
            "a": [["101", "0"], ["103", "1"]],
        },
        generation=generation,
        received_at_ms=now,
    )
    await operator.ingest_oracle(
        ReferencePriceSample(
            timestamp_ms=now,
            received_at_ms=now,
            price=100.0,
            source="polymarket_rtds_chainlink:btc/usd",
        ),
        generation=generation,
    )
    base = {
        "event_type": "book",
        "sequence": 1,
        "timestamp": now,
        "bids": [{"price": "0.09", "size": "100"}],
        "asks": [{"price": "0.10", "size": "100"}],
    }
    market = operator.active_market
    assert market is not None
    await operator.ingest_market_message(
        {**base, "asset_id": market.yes_token_id}, generation=generation
    )
    if not produce_decision:
        return None
    return await operator.ingest_market_message(
        {
            **base,
            "asset_id": market.no_token_id,
            "bids": [{"price": "0.89", "size": "100"}],
            "asks": [{"price": "0.90", "size": "100"}],
        },
        generation=generation,
    )


async def _emit_market_pair(
    operator: PaperTradingOperator,
    market: DiscoveredMarket,
    *,
    timestamp_ms: int,
    sequence: int,
) -> object | None:
    base = {
        "event_type": "book",
        "sequence": sequence,
        "timestamp": timestamp_ms,
        "bids": [{"price": "0.09", "size": "100"}],
        "asks": [{"price": "0.10", "size": "100"}],
    }
    await operator.ingest_market_message(
        {**base, "asset_id": market.yes_token_id},
        generation=operator.generation,
    )
    return await operator.ingest_market_message(
        {
            **base,
            "asset_id": market.no_token_id,
            "bids": [{"price": "0.89", "size": "100"}],
            "asks": [{"price": "0.90", "size": "100"}],
        },
        generation=operator.generation,
    )


async def test_start_creates_stable_session_and_restart_resumes_same_run(tmp_path: Path) -> None:
    market = _market(1)
    selection = _selection(market)
    clock = FakeClock(10_000)
    first = PaperTradingOperator(
        config=_config(tmp_path),
        discovery=FakeDiscovery([selection]),
        resolution=FakeResolution([None]),
        clock_ms=clock,
    )
    await first.start()
    assert first.state is OperatorState.SYNCING
    assert first.session is not None
    first_run_id = first.run_id
    assert first_run_id == stable_run_id(
        strategy_id="strategy-test",
        market_id=market.market_id,
        window_id=market.window_id,
        paper_account_id="paper-account-test",
    )
    await _ready_operator(first, clock)
    first_snapshot = first.session.current_snapshot
    await first.shutdown()

    second = PaperTradingOperator(
        config=_config(tmp_path),
        discovery=FakeDiscovery([selection]),
        resolution=FakeResolution([None]),
        clock_ms=clock,
    )
    await second.start()
    assert second.run_id == first_run_id
    assert second.session is not None
    assert second.session.current_snapshot == first_snapshot
    assert second.counters["fills"] == 1


async def test_freshness_gate_blocks_until_all_three_sources_ready(tmp_path: Path) -> None:
    market = _market(1)
    clock = FakeClock(10_000)
    operator = PaperTradingOperator(
        config=_config(tmp_path),
        discovery=FakeDiscovery([_selection(market)]),
        resolution=FakeResolution([None]),
        clock_ms=clock,
    )
    await operator.start()
    generation = operator.generation
    base = {
        "event_type": "book",
        "sequence": 1,
        "timestamp": clock.now_ms,
        "bids": [{"price": "0.4", "size": "5"}],
        "asks": [{"price": "0.5", "size": "5"}],
    }
    await operator.ingest_market_message({**base, "asset_id": market.yes_token_id}, generation=generation)
    await operator.ingest_market_message({**base, "asset_id": market.no_token_id}, generation=generation)

    assert operator.state is OperatorState.SYNCING
    assert operator.session is not None
    assert operator.session.runner.oms_calls == 0
    assert operator.counters["snapshot_freshness_dropped"] == 1


@pytest.mark.parametrize("settle_before_restart", [True, False])
async def test_restart_recovers_account_frontier_before_discovering_new_market(
    tmp_path, settle_before_restart,
) -> None:
    old, new = _market(1), _market(2, start=900_000)
    clock = FakeClock(10_000)
    first = PaperTradingOperator(
        config=_config(tmp_path),
        discovery=FakeDiscovery([_selection(old), DiscoverySelection(None, None, 0)]),
        resolution=FakeResolution([_final(old)]), clock_ms=clock,
    )
    await first.start()
    await _ready_operator(first, clock)
    assert first.session is not None
    assert first.session.current_snapshot.open_lots
    clock.now_ms = old.end_ts_ms + 1
    if settle_before_restart:
        await first.poll()
        assert first.active_market == old
    await first.shutdown()

    discovery = FakeDiscovery([_selection(new)])
    resolution = FakeResolution([_final(old)])
    second = PaperTradingOperator(
        config=_config(tmp_path), discovery=discovery, resolution=resolution, clock_ms=clock,
    )
    await second.start()
    assert second.active_market == old
    assert discovery.calls == 0
    assert second.session is not None
    old_session = second.session
    await second.poll()
    carried_cash = old_session.current_snapshot.cash
    assert carried_cash > second.config.initial_bankroll
    assert second.active_market == new
    assert second.session.store.manifest.initial_bankroll == carried_cash
    assert second.session.current_snapshot.cash == carried_cash
    assert resolution.calls == (0 if settle_before_restart else 1)
    checkpoint = second.checkpoint_store.load(config_sha256=second.config.config_sha256)
    assert checkpoint.predecessor_run_id == old_session.store.manifest.run_id
    assert checkpoint.predecessor_settled_cash == carried_cash
    await second.shutdown()
    third = PaperTradingOperator(
        config=_config(tmp_path), discovery=discovery, resolution=resolution, clock_ms=clock,
    )
    await third.start()
    assert third.active_market == new
    assert third.session.current_snapshot.cash == carried_cash


@pytest.mark.parametrize("failure_boundary", ["checkpoint", "create_session"])
async def test_rollover_crash_recovers_old_ledger_or_successor_activation_intent(
    tmp_path, monkeypatch, failure_boundary,
) -> None:
    old, new = _market(1), _market(2, start=900_000)
    clock = FakeClock(10_000)
    first = PaperTradingOperator(
        config=_config(tmp_path), discovery=FakeDiscovery([_selection(old), _selection(new)]),
        resolution=FakeResolution([_final(old)]), clock_ms=clock,
    )
    await first.start()
    await _ready_operator(first, clock)
    old_session = first.session
    assert old_session is not None

    def fail(*_args, **_kwargs):
        raise OSError("injected crash boundary")

    with monkeypatch.context() as patch:
        if failure_boundary == "checkpoint":
            patch.setattr(first.checkpoint_store, "write", fail)
        else:
            patch.setattr(PaperTradingSession, "create_new", fail)
        clock.now_ms = old.end_ts_ms + 1
        await first.poll()
    assert first.state is OperatorState.FAILED
    carried_cash = old_session.current_snapshot.cash
    assert carried_cash > first.config.initial_bankroll
    await first.shutdown()  # simulate process exit releasing the OS ownership lock

    second = PaperTradingOperator(
        config=_config(tmp_path), discovery=FakeDiscovery([_selection(new)]),
        resolution=FakeResolution([None]), clock_ms=clock,
    )
    await second.start()
    if failure_boundary == "checkpoint":
        assert second.active_market == old
        await second.poll()
    assert second.active_market == new
    assert second.session is not None
    assert second.session.store.manifest.initial_bankroll == carried_cash
    assert second.session.current_snapshot.cash == carried_cash


@pytest.mark.parametrize("checkpoint_damage", ["missing", "corrupt"])
async def test_missing_or_corrupt_checkpoint_never_resets_existing_account(
    tmp_path, checkpoint_damage,
) -> None:
    clock = FakeClock(10_000)
    first = PaperTradingOperator(
        config=_config(tmp_path), discovery=FakeDiscovery([_selection(_market(1))]),
        resolution=FakeResolution([None]), clock_ms=clock,
    )
    await first.start()
    await first.shutdown()
    if checkpoint_damage == "missing":
        first.checkpoint_store.path.unlink()
    else:
        first.checkpoint_store.path.write_text("{broken")
    second = PaperTradingOperator(
        config=_config(tmp_path), discovery=FakeDiscovery([_selection(_market(2, start=900_000))]),
        resolution=FakeResolution([None]), clock_ms=clock,
    )
    await second.start()
    assert second.state is OperatorState.FAILED
    assert second.session is None
    assert len(list(tmp_path.glob("paper-*"))) == 1


async def test_rejected_spot_invalidates_alpha_and_blocks_oms(tmp_path) -> None:
    market, clock = _market(1), FakeClock(10_000)
    operator = PaperTradingOperator(
        config=_config(tmp_path), discovery=FakeDiscovery([_selection(market)]),
        resolution=FakeResolution([None]), clock_ms=clock,
    )
    await operator.start()
    await _ready_operator(operator, clock)
    assert operator.session is not None
    oms_calls = operator.session.runner.oms_calls
    clock.now_ms += 1
    accepted = await operator.ingest_binance_delta(
        {"s": "BTCUSDT", "E": clock.now_ms, "U": 12, "u": 12,
         "b": [["201", "2"]], "a": [["103", "0"], ["203", "2"]]},
        generation=operator.generation, received_at_ms=clock.now_ms,
    )
    assert accepted is False
    assert operator.pricing_provider.outlier_count == 1
    assert operator.pricing_provider.last_spot_timestamp_ms is None
    assert operator.session.runner.ofi_engine.last_timestamp_ms is None
    await _emit_market_pair(operator, market, timestamp_ms=clock.now_ms, sequence=2)
    assert operator.state is not OperatorState.RUNNING
    assert operator.session.runner.oms_calls == oms_calls


async def test_live_connection_rejects_spot_and_requests_new_bootstrap(tmp_path) -> None:
    market, clock = _market(1), FakeClock(10_000)
    operator = PaperTradingOperator(
        config=_config(tmp_path), discovery=FakeDiscovery([_selection(market)]),
        resolution=FakeResolution([None]), clock_ms=clock,
    )
    await operator.start()
    assert await operator.begin_binance_connection(
        window_generation=operator.generation, connection_generation=1,
        snapshot={"lastUpdateId": 10, "bids": [["99", "2"]], "asks": [["101", "2"]]},
        received_at_ms=clock.now_ms,
    )
    assert await operator.ingest_binance_connection_delta(
        {"s": "BTCUSDT", "E": clock.now_ms, "U": 11, "u": 11, "b": [], "a": []},
        window_generation=operator.generation, connection_generation=1,
        received_at_ms=clock.now_ms,
    )
    clock.now_ms += 1
    assert not await operator.ingest_binance_connection_delta(
        {"s": "BTCUSDT", "E": clock.now_ms, "U": 12, "u": 12,
         "b": [["201", "2"]], "a": [["101", "0"], ["203", "2"]]},
        window_generation=operator.generation, connection_generation=1,
        received_at_ms=clock.now_ms,
    )
    assert operator.binance_sync.needs_bootstrap
    assert operator.pricing_provider.last_spot_timestamp_ms is None
    assert operator.session.runner.ofi_engine.last_timestamp_ms is None
    assert operator.session.runner.oms_calls == 0


async def test_projection_failure_closes_execution_gate_and_recovers(tmp_path) -> None:
    market, clock = _market(1), FakeClock(10_000)
    operator = PaperTradingOperator(
        config=_config(tmp_path), discovery=FakeDiscovery([_selection(market)]),
        resolution=FakeResolution([None]), clock_ms=clock,
    )
    await operator.start()
    await _ready_operator(operator, clock)
    session = operator.session
    assert session is not None
    calls = session.runner.oms_calls
    account = session.current_snapshot
    working_writer = operator.status_writer
    operator.status_writer = FailingStatusWriter(working_writer.path)
    clock.now_ms += operator.config.status_interval_ms
    await operator.poll()
    assert operator.state is OperatorState.DEGRADED
    await _emit_market_pair(operator, market, timestamp_ms=clock.now_ms, sequence=2)
    assert operator.state is OperatorState.DEGRADED
    assert session.runner.oms_calls == calls
    assert session.current_snapshot == account
    operator.status_writer = working_writer
    clock.now_ms += 1
    await _emit_market_pair(operator, market, timestamp_ms=clock.now_ms, sequence=3)
    assert operator.state is OperatorState.RUNNING
    assert session.runner.oms_calls > calls


async def test_deep_binance_updates_do_not_refresh_stale_ofi_for_oms(
    tmp_path: Path,
) -> None:
    market = _market(1)
    clock = FakeClock(10_000)
    config = replace(
        _config(tmp_path),
        max_alpha_age_ms=2_000,
        max_pricing_age_ms=5_000,
    )
    operator = PaperTradingOperator(
        config=config,
        discovery=FakeDiscovery([_selection(market)]),
        resolution=FakeResolution([None]),
        clock_ms=clock,
    )
    await operator.start()
    await _ready_operator(operator, clock)
    assert operator.session is not None
    oms_calls = operator.session.runner.oms_calls

    clock.now_ms = 13_001
    assert await operator.ingest_binance_delta(
        {
            "s": "BTCUSDT",
            "E": clock.now_ms,
            "U": 12,
            "u": 12,
            "b": [["90", "1"]],
            "a": [],
        },
        generation=operator.generation,
        received_at_ms=clock.now_ms,
    )
    await operator.ingest_oracle(
        ReferencePriceSample(
            timestamp_ms=clock.now_ms,
            received_at_ms=clock.now_ms,
            price=100.0,
            source="polymarket_rtds_chainlink:btc/usd",
        ),
        generation=operator.generation,
    )
    await _emit_market_pair(
        operator,
        market,
        timestamp_ms=clock.now_ms,
        sequence=2,
    )

    status = operator.status()
    assert status.feeds["binance"]["fresh"] is True
    assert status.alpha["fresh"] is False
    assert operator.state is OperatorState.SYNCING
    assert operator.session.runner.oms_calls == oms_calls


async def test_chainlink_disconnect_immediately_closes_oms_gate(tmp_path: Path) -> None:
    market = _market(1)
    clock = FakeClock(10_000)
    operator = PaperTradingOperator(
        config=_config(tmp_path),
        discovery=FakeDiscovery([_selection(market)]),
        resolution=FakeResolution([None]),
        clock_ms=clock,
    )
    await operator.start()
    await _ready_operator(operator, clock)
    assert operator.session is not None
    oms_calls = operator.session.runner.oms_calls

    await operator.disconnect_feed("chainlink", window_generation=operator.generation)
    assert operator.state is OperatorState.SYNCING
    assert operator.status().feeds["chainlink"]["connected"] is False

    clock.now_ms += 1
    await _emit_market_pair(
        operator,
        market,
        timestamp_ms=clock.now_ms,
        sequence=2,
    )
    assert operator.session.runner.oms_calls == oms_calls


async def test_restart_reconnect_replay_is_disk_deduplicated(tmp_path: Path) -> None:
    market = _market(1)
    clock = FakeClock(10_000)
    selection = _selection(market)
    first = PaperTradingOperator(
        config=_config(tmp_path),
        discovery=FakeDiscovery([selection]),
        resolution=FakeResolution([None]),
        clock_ms=clock,
    )
    await first.start()
    await _ready_operator(first, clock)
    assert first.session is not None
    cash_after_fill = first.session.current_snapshot.cash
    await first.shutdown()

    second = PaperTradingOperator(
        config=_config(tmp_path),
        discovery=FakeDiscovery([selection]),
        resolution=FakeResolution([None]),
        clock_ms=clock,
    )
    await second.start()
    await _ready_operator(second, clock)
    assert second.session is not None
    assert second.session.current_snapshot.cash == cash_after_fill
    assert second.counters["fills"] == 1
    assert second.counters["snapshot_deduplicated"] == 1


async def test_expiry_pending_final_once_and_rollover_fences_old_generation(tmp_path: Path) -> None:
    first_market = _market(1)
    second_market = _market(2, start=first_market.end_ts_ms)
    clock = FakeClock(10_000)
    discovery = FakeDiscovery(
        [_selection(first_market, second_market), _selection(second_market)]
    )
    resolution = FakeResolution([None, _final(first_market), _final(first_market)])
    operator = PaperTradingOperator(
        config=_config(tmp_path),
        discovery=discovery,
        resolution=resolution,
        clock_ms=clock,
    )
    await operator.start()
    old_generation = operator.generation
    await _ready_operator(operator, clock)

    clock.now_ms = first_market.end_ts_ms
    await operator.poll()
    assert operator.state is OperatorState.SETTLEMENT_PENDING
    assert operator.counters["settlement_completed"] == 0
    await operator.poll()
    assert operator.active_market == second_market
    assert operator.counters["settlement_completed"] == 1
    assert operator.counters["rollovers"] == 1
    assert operator.generation == old_generation + 1
    assert operator.status().settlement == {
        "status": "OPEN",
        "source_reference": None,
    }

    accepted = await operator.ingest_binance_delta(
        {"s": "BTCUSDT", "E": clock.now_ms, "U": 1, "u": 1, "b": [], "a": []},
        generation=old_generation,
    )
    assert accepted is False
    assert operator.counters["snapshot_generation_dropped"] == 1


async def test_no_decision_after_expiry_and_settlement_replay_does_not_duplicate(tmp_path: Path) -> None:
    market = _market(1)
    clock = FakeClock(10_000)
    selection = _selection(market)
    operator = PaperTradingOperator(
        config=_config(tmp_path),
        discovery=FakeDiscovery([selection, selection]),
        resolution=FakeResolution([_final(market), _final(market)]),
        clock_ms=clock,
    )
    await operator.start()
    await _ready_operator(operator, clock)
    assert operator.session is not None
    oms_calls = operator.session.runner.oms_calls
    clock.now_ms = market.end_ts_ms
    await _ready_operator(operator, clock)
    assert operator.session.runner.oms_calls == oms_calls
    await operator.poll()
    await operator.poll()
    assert operator.counters["settlement_completed"] == 1
    assert len(operator.session.store.recent_settlements(limit=2)) == 1


async def test_settled_window_retries_when_next_market_is_temporarily_unavailable(
    tmp_path: Path,
) -> None:
    market = _market(1)
    clock = FakeClock(10_000)

    class TemporarilyUnavailableDiscovery:
        def __init__(self) -> None:
            self.calls = 0

        async def discover(self, **_kwargs: object) -> DiscoverySelection:
            self.calls += 1
            if self.calls == 1:
                return _selection(market)
            raise RuntimeError("next market is not published yet")

    operator = PaperTradingOperator(
        config=_config(tmp_path),
        discovery=TemporarilyUnavailableDiscovery(),
        resolution=FakeResolution([_final(market)]),
        clock_ms=clock,
    )
    await operator.start()
    clock.now_ms = market.end_ts_ms + 1

    await operator.poll()
    assert operator.state is OperatorState.DEGRADED
    assert operator.counters["settlement_completed"] == 1
    await operator.poll()
    assert operator.state is OperatorState.DEGRADED
    assert operator.state_reason.startswith("rollover_discovery_unavailable")
    assert operator.counters["settlement_completed"] == 1


async def test_persistence_failure_is_permanent_fail_closed(tmp_path: Path, monkeypatch) -> None:
    market = _market(1)
    clock = FakeClock(10_000)
    operator = PaperTradingOperator(
        config=_config(tmp_path),
        discovery=FakeDiscovery([_selection(market)]),
        resolution=FakeResolution([None]),
        clock_ms=clock,
    )
    await operator.start()
    assert operator.session is not None

    def fail_append(**_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(operator.session.store, "append_decision", fail_append)
    with pytest.raises(PaperSessionFailedError):
        await _ready_operator(operator, clock)
    assert operator.state is OperatorState.FAILED
    assert operator.session.failed
    await operator.poll()
    assert operator.state is OperatorState.FAILED


async def test_shutdown_waits_for_inflight_decision(tmp_path: Path, monkeypatch) -> None:
    market = _market(1)
    clock = FakeClock(10_000)
    operator = PaperTradingOperator(
        config=_config(tmp_path),
        discovery=FakeDiscovery([_selection(market)]),
        resolution=FakeResolution([None]),
        clock_ms=clock,
    )
    await operator.start()
    await _ready_operator(operator, clock, produce_decision=False)
    entered = asyncio.Event()
    release = asyncio.Event()
    original = PaperTradingSession.process_snapshot

    async def blocked(self: PaperTradingSession, snapshot):
        entered.set()
        await release.wait()
        return await original(self, snapshot)

    monkeypatch.setattr(PaperTradingSession, "process_snapshot", blocked)
    market_sync = operator.market_sync
    assert market_sync is not None
    snapshot = market_sync.ingest(
        {
            "event_type": "book",
            "sequence": 1,
            "timestamp": clock.now_ms,
            "asset_id": market.no_token_id,
            "bids": [{"price": "0.89", "size": "100"}],
            "asks": [{"price": "0.90", "size": "100"}],
        },
        generation=operator.generation,
    )
    assert snapshot is not None
    process_task = asyncio.create_task(operator.process_snapshot(snapshot, generation=operator.generation))
    await entered.wait()
    shutdown_task = asyncio.create_task(operator.shutdown())
    done, _pending = await asyncio.wait({shutdown_task}, timeout=0)
    assert not done
    release.set()
    await process_task
    await shutdown_task
    assert operator.state is OperatorState.STOPPED


async def test_missing_authoritative_strike_degrades_without_session(tmp_path: Path) -> None:
    market = replace(_market(1), reference_price_at_start=None)
    operator = PaperTradingOperator(
        config=_config(tmp_path),
        discovery=FakeDiscovery([_selection(market)]),
        resolution=FakeResolution([None]),
        clock_ms=FakeClock(10_000),
    )
    await operator.start()
    assert operator.state is OperatorState.DEGRADED
    assert operator.session is None


async def test_manifest_config_mismatch_is_permanent_failed_on_resume(tmp_path: Path) -> None:
    market = _market(1)
    selection = _selection(market)
    clock = FakeClock(10_000)
    first = PaperTradingOperator(
        config=_config(tmp_path),
        discovery=FakeDiscovery([selection]),
        resolution=FakeResolution([None]),
        clock_ms=clock,
    )
    await first.start()
    await first.shutdown()
    assert first.session is not None

    changed = replace(_config(tmp_path), pricing_ofi_gamma=0.5)
    second = PaperTradingOperator(
        config=changed,
        discovery=FakeDiscovery([selection]),
        resolution=FakeResolution([None]),
        clock_ms=clock,
    )
    await second.start()
    assert second.state is OperatorState.FAILED
    assert second.session is None
    with pytest.raises(RuntimeError, match="permanently failed"):
        await second.start()
    assert second.state is OperatorState.FAILED


async def test_projection_failure_degrades_but_does_not_damage_ledger(tmp_path: Path) -> None:
    market = _market(1)
    operator = PaperTradingOperator(
        config=_config(tmp_path),
        discovery=FakeDiscovery([_selection(market)]),
        resolution=FakeResolution([None]),
        clock_ms=FakeClock(10_000),
        status_writer=FailingStatusWriter(tmp_path / "status.json"),
    )
    await operator.start()
    assert operator.state is OperatorState.DEGRADED
    assert operator.state_reason == "status_projection_write_failed"
    assert operator.counters["projection_errors"] > 0
    assert operator.session is not None
    recovered = operator.session.store.recover_ledger()
    assert recovered.snapshot() == operator.session.current_snapshot


async def test_second_operator_cannot_read_or_write_owned_account(tmp_path, monkeypatch):
    clock = FakeClock(10_000)
    first = PaperTradingOperator(
        config=_config(tmp_path), discovery=FakeDiscovery([_selection(_market(1))]),
        resolution=FakeResolution([None]), clock_ms=clock,
    )
    await first.start()
    await _ready_operator(first, clock)
    checkpoint_bytes = first.checkpoint_store.path.read_bytes()
    status_bytes = first.status_writer.path.read_bytes()
    before = first.session.store.recover_ledger().snapshot()
    second = PaperTradingOperator(
        config=_config(tmp_path), discovery=FakeDiscovery([_selection(_market(1))]),
        resolution=FakeResolution([None]), clock_ms=clock,
    )

    def forbidden(**_kwargs):
        pytest.fail("contender must not even read the account checkpoint")

    with monkeypatch.context() as patch:
        patch.setattr(second.checkpoint_store, "load", forbidden)
        with pytest.raises(AccountOwnershipError):
            await second.start()
        await second.shutdown()
    assert first.checkpoint_store.path.read_bytes() == checkpoint_bytes
    assert first.status_writer.path.read_bytes() == status_bytes
    assert first.session.store.recover_ledger().snapshot() == before
    await first.shutdown()
    await second.start()
    assert second.session.current_snapshot == before
    await second.shutdown()


async def test_missing_active_run_is_never_recreated(tmp_path):
    clock = FakeClock(10_000)
    first = PaperTradingOperator(
        config=_config(tmp_path), discovery=FakeDiscovery([_selection(_market(1))]),
        resolution=FakeResolution([None]), clock_ms=clock,
    )
    await first.start()
    await _ready_operator(first, clock)
    checkpoint = first.checkpoint_store.load(config_sha256=first.config.config_sha256)
    assert checkpoint.activation_state == "ACTIVE"
    run_path = first.session.store.run_dir
    await first.shutdown()
    run_path.rename(tmp_path / "retained-original-run")
    second = PaperTradingOperator(
        config=_config(tmp_path), discovery=FakeDiscovery([_selection(_market(1))]),
        resolution=FakeResolution([None]), clock_ms=clock,
    )
    await second.start()
    assert second.state is OperatorState.FAILED
    assert second.session is None
    assert not run_path.exists()
    assert second.checkpoint_store.load(config_sha256=second.config.config_sha256) == checkpoint
    await second.shutdown()


async def test_created_run_recovers_if_active_checkpoint_publication_crashes(tmp_path, monkeypatch):
    clock = FakeClock(10_000)
    first = PaperTradingOperator(
        config=_config(tmp_path), discovery=FakeDiscovery([_selection(_market(1))]),
        resolution=FakeResolution([None]), clock_ms=clock,
    )
    original_write = first.checkpoint_store.write

    def fail_active(checkpoint):
        if checkpoint.activation_state == "ACTIVE":
            raise OSError("crash before ACTIVE publication")
        original_write(checkpoint)

    monkeypatch.setattr(first.checkpoint_store, "write", fail_active)
    await first.start()
    assert first.state is OperatorState.FAILED
    intent = first.checkpoint_store.load(config_sha256=first.config.config_sha256)
    assert intent.activation_state == "ACTIVATING"
    assert (tmp_path / intent.run_id).is_dir()
    assert first.session is None  # not exposed before ACTIVE is durable
    await first.shutdown()
    second = PaperTradingOperator(
        config=_config(tmp_path), discovery=FakeDiscovery([_selection(_market(1))]),
        resolution=FakeResolution([None]), clock_ms=clock,
    )
    await second.start()
    assert second.state is OperatorState.SYNCING
    assert second.checkpoint_store.load(config_sha256=second.config.config_sha256).activation_state == "ACTIVE"
    assert second.session.current_snapshot.last_event_sequence == 0
    await second.shutdown()


@pytest.mark.parametrize("first_delta_changes_top", [True, False])
async def test_buffered_delta_before_rest_receipt_seeds_both_event_time_series(
    tmp_path, first_delta_changes_top,
):
    clock = FakeClock(1_200)
    operator = PaperTradingOperator(
        config=_config(tmp_path), discovery=FakeDiscovery([_selection(_market(1))]),
        resolution=FakeResolution([None]), clock_ms=clock,
    )
    await operator.start()
    assert await operator.begin_binance_connection(
        window_generation=operator.generation, connection_generation=1, received_at_ms=1_200,
        snapshot={"lastUpdateId": 10, "bids": [["99", "2"]], "asks": [["101", "2"]]},
    )
    assert operator.pricing_provider.last_spot_timestamp_ms is None
    assert operator.session.runner.ofi_engine.last_timestamp_ms is None
    assert await operator.ingest_binance_connection_delta(
        {"s": "BTCUSDT", "E": 1_100, "U": 11, "u": 11,
         "b": [["99", "3"]] if first_delta_changes_top else [], "a": []},
        window_generation=operator.generation, connection_generation=1, received_at_ms=1_150,
    )
    assert not operator.binance_sync.needs_bootstrap
    assert operator.pricing_provider.last_spot_timestamp_ms == 1_100
    assert operator.session.runner.ofi_engine.last_timestamp_ms == 1_100
    assert operator.pricing_provider.return_sample_count == 0
    assert await operator.ingest_binance_connection_delta(
        {"s": "BTCUSDT", "E": 1_101, "U": 12, "u": 12, "b": [["100", "3"]], "a": []},
        window_generation=operator.generation, connection_generation=1, received_at_ms=1_151,
    )
    assert operator.pricing_provider.return_sample_count == 1
    assert operator.pricing_provider.out_of_order_count == 0
    await operator.shutdown()


async def test_account_totals_and_bounded_history_survive_three_windows_and_restart(tmp_path):
    markets = [_market(index + 1, start=index * 900_000) for index in range(3)]
    clock = FakeClock(10_000)
    config = replace(_config(tmp_path), fee_bps=100)
    operator = PaperTradingOperator(
        config=config, discovery=FakeDiscovery([_selection(market) for market in markets]),
        resolution=FakeResolution([_final(market) for market in markets]), clock_ms=clock,
    )
    await operator.start()
    settled_runs, pnl, fees = [], 0.0, 0.0
    for market in markets[:2]:
        clock.now_ms = market.start_ts_ms + 10_000
        await _ready_operator(operator, clock)
        old_session = operator.session
        settled_runs.append(operator.run_id)
        clock.now_ms = market.end_ts_ms + 1
        await operator.poll()
        pnl += old_session.current_snapshot.realized_pnl
        fees += old_session.current_snapshot.commission_paid
    status = operator.status()
    assert status.account["initial_bankroll"] == 1_000
    assert status.account["realized_pnl"] == pytest.approx(pnl)
    assert status.account["total_fees"] == pytest.approx(fees)
    assert status.account["current_run_realized_pnl"] == 0
    assert status.account["current_run_initial_bankroll"] == pytest.approx(1_000 + pnl)
    assert operator.read_repository.account_summary()["realized_pnl"] == pytest.approx(pnl)
    assert [row["run_id"] for row in operator.read_repository.recent_fills(2)] == settled_runs
    assert [row["run_id"] for row in operator.read_repository.settlements(2)] == settled_runs
    assert len(operator.read_repository.recent_decisions(1)) == 1
    assert [row["run_id"] for row in operator.read_repository.recent_runs(3)] == [operator.run_id, *reversed(settled_runs)]
    assert operator.read_repository.recent_fills(1, before_run_id=settled_runs[-1])[0]["run_id"] == settled_runs[0]
    one_run_page = OperatorReadRepository(
        status_path=operator.status_writer.path, run_store=operator.session.store,
        checkpoint=operator._checkpoint, default_limit=1, max_limit=1,
    )
    assert one_run_page.recent_fills() == ()  # scan is bounded even across empty runs
    assert one_run_page.recent_fills(before_run_id=operator.run_id)[0]["run_id"] == settled_runs[-1]
    cash = operator.session.current_snapshot.cash
    await operator.shutdown()
    resumed = PaperTradingOperator(
        config=config, discovery=FakeDiscovery([_selection(markets[-1])]),
        resolution=FakeResolution([None]), clock_ms=clock,
    )
    await resumed.start()
    assert resumed.session.current_snapshot.cash == cash
    assert resumed.status().account["realized_pnl"] == pytest.approx(pnl)
    assert resumed.status().account["total_fees"] == pytest.approx(fees)
    assert [row["run_id"] for row in resumed.read_repository.recent_fills(2)] == settled_runs
    await resumed.shutdown()


async def test_zero_cash_settlement_is_exhausted_not_storage_failure(tmp_path, monkeypatch):
    market, clock = _market(1), FakeClock(10_000)
    config = replace(
        _config(tmp_path), initial_bankroll=10, max_single_trade_pct=1,
        max_position_pct=1, max_window_exposure_pct=1, slippage_tolerance=0,
    )
    original_evaluate = PolymarketPricingEngine.evaluate_signal

    def all_in(self, **kwargs):
        signal = original_evaluate(self, **kwargs)
        return replace(signal, direction=SignalDirection.BUY_YES, recommended_size_pct=1)

    monkeypatch.setattr(PolymarketPricingEngine, "evaluate_signal", all_in)
    discovery = FakeDiscovery([_selection(market), _selection(_market(2, start=900_000))])
    operator = PaperTradingOperator(
        config=config, discovery=discovery, resolution=FakeResolution([replace(_final(market), yes_payout=0)]),
        clock_ms=clock,
    )
    await operator.start()
    await _ready_operator(operator, clock)
    assert operator.session.current_snapshot.cash == 0
    clock.now_ms = market.end_ts_ms + 1
    await operator.poll()
    assert operator.state is OperatorState.EXHAUSTED
    assert operator.session.failed is False
    assert operator.status().account["realized_pnl"] == -10
    assert discovery.calls == 1
    calls = operator.session.runner.oms_calls
    await operator.poll()
    await _emit_market_pair(operator, market, timestamp_ms=clock.now_ms, sequence=2)
    assert operator.session.runner.oms_calls == calls
    assert len(list(tmp_path.glob("paper-*"))) == 1
    await operator.shutdown()
    assert operator.state is OperatorState.EXHAUSTED
    resumed = PaperTradingOperator(
        config=config, discovery=discovery, resolution=FakeResolution([None]), clock_ms=clock,
    )
    await resumed.start()
    assert resumed.state is OperatorState.EXHAUSTED
    assert resumed.session.current_snapshot.cash == 0
    assert resumed.read_repository.settlements(1)
    await resumed.shutdown()
