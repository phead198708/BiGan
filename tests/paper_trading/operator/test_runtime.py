from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from bigan.paper_trading.operator.config import OperatorConfig
from bigan.paper_trading.operator.discovery import DiscoveredMarket, DiscoverySelection
from bigan.paper_trading.operator.pricing_inputs import ReferencePriceSample
from bigan.paper_trading.operator.read_model import OperatorState, OperatorStatusWriter
from bigan.paper_trading.operator.resolution import FinalResolution
from bigan.paper_trading.operator.runtime import PaperTradingOperator, stable_run_id
from bigan.paper_trading.session import PaperSessionFailedError, PaperTradingSession


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
        {"lastUpdateId": 10, "bids": [["99", "2"]], "asks": [["101", "2"]]},
        generation=generation,
        received_at_ms=now - 1,
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
