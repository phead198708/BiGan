"""Fixed-window StrategyRunner → ledger → storage integration tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from bigan.data.polymarket_clob import MarketSnapshot, PolymarketFeedHandler
from bigan.execution.polymarket_oms import PolymarketOMS
from bigan.features.binance_ofi import BinanceOFICalculator, TopOfBook
from bigan.paper_trading.contracts import PaperSettlementInput
from bigan.paper_trading.session import PaperSessionFailedError, PaperTradingSession
from bigan.paper_trading.storage import (
    LEDGER_EVENTS_FILE,
    SETTLEMENT_EVENTS_FILE,
    SIGNAL_EVENTS_FILE,
)
from bigan.pipeline.events import DecisionDisposition
from bigan.pipeline.strategy_runner import PricingInputs, StrategyRunner
from bigan.strategies.polymarket_pricing import MarketWindow, PolymarketPricingEngine

WINDOW = MarketWindow(
    window_id="btc-paper-window",
    symbol="BTC",
    strike_price=100_000.0,
    start_ts_ms=0,
    end_ts_ms=900_000,
    window_type="15m",
)


def _runner(
    *,
    fee_bps: float = 100.0,
    ema_alpha: float = 1.0,
    zscore_min_samples: int = 2,
    spot_price: float = 100_000.0,
    oms_symbol: str = "BTC",
    provider_identity: str | None = "fixed-pricing-inputs-v1",
    signal_cache_size: int = 100_000,
) -> StrategyRunner:
    return StrategyRunner(
        ofi_engine=BinanceOFICalculator(
            zscore_min_samples=zscore_min_samples,
            ema_alpha=ema_alpha,
        ),
        pricing_engine=PolymarketPricingEngine(),
        oms=PolymarketOMS(
            max_spread_allowed=0.08,
            symbol=oms_symbol,
            signal_cache_size=signal_cache_size,
        ),
        feed_handler=PolymarketFeedHandler(window_id=WINDOW.window_id, mock=True),
        window=WINDOW,
        initial_bankroll=1_000.0,
        spot_price=spot_price,
        pricing_inputs_provider=lambda timestamp_ms: PricingInputs(
            timestamp_ms=timestamp_ms,
            spot_price=spot_price,
            oracle_twap_so_far=WINDOW.strike_price,
            twap_weight=0.0,
            volatility_annualized=0.60,
        ),
        fee_bps=fee_bps,
        pricing_inputs_provider_identity=provider_identity,
    )


def _snapshot(
    timestamp_ms: int,
    *,
    yes_bid: float,
    yes_ask: float,
    no_bid: float,
    no_ask: float,
) -> MarketSnapshot:
    return MarketSnapshot(
        timestamp_ms=timestamp_ms,
        window_id=WINDOW.window_id,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=no_bid,
        no_ask=no_ask,
        last_traded_price=yes_ask,
        yes_bid_size=100.0,
        yes_ask_size=100.0,
        no_bid_size=100.0,
        no_ask_size=100.0,
    )


def _alpha(timestamp_ms: int, price: float) -> TopOfBook:
    return TopOfBook(
        ts_ms=timestamp_ms,
        bid_price=price,
        bid_qty=10.0,
        ask_price=price + 1.0,
        ask_qty=10.0,
    )


def _settlement() -> PaperSettlementInput:
    return PaperSettlementInput(
        window_id=WINDOW.window_id,
        yes_payout=1.0,
        settlement_ts_ms=WINDOW.end_ts_ms,
        source="deterministic-test-oracle",
        source_ts_ms=WINDOW.end_ts_ms,
        received_ts_ms=WINDOW.end_ts_ms + 1,
        source_reference="fixture://btc-paper-window",
    )


def test_fixed_window_e2e_resume_and_settlement(tmp_path: Path) -> None:
    runner = _runner()
    session = PaperTradingSession.create_new(
        runner=runner,
        output_dir=tmp_path,
        run_id="paper-e2e",
        source_commit="fbd64d9",
        config={"feed": "offline-fixture"},
        created_at="2026-09-03T00:00:00+00:00",
    )

    runner.push_alpha_tick(_alpha(100_000, 100_000.0))
    session.process_snapshot_sync(
        _snapshot(100_000, yes_bid=0.98, yes_ask=0.99, no_bid=0.0, no_ask=0.99)
    )
    runner.push_alpha_tick(_alpha(100_100, 100_010.0))
    fill_snapshot = replace(
        _snapshot(100_100, yes_bid=0.39, yes_ask=0.40, no_bid=0.09, no_ask=0.90),
        yes_ask_size=10.0,
    )
    fill = session.process_snapshot_sync(fill_snapshot)
    runner.push_alpha_tick(_alpha(100_200, 100_020.0))
    rejection = session.process_snapshot_sync(
        _snapshot(100_200, yes_bid=0.20, yes_ask=0.40, no_bid=0.09, no_ask=0.90)
    )

    assert fill is not None and fill.status == "FILLED"
    assert rejection is not None and rejection.status == "REJECTED"
    before_resume = session.current_snapshot
    assert before_resume.cash == pytest.approx(runner.current_bankroll)
    assert before_resume.cash == pytest.approx(runner.oms.bankroll)
    assert len(before_resume.open_lots) == 1

    rows = [
        json.loads(line)
        for line in (session.store.run_dir / SIGNAL_EVENTS_FILE).read_text().splitlines()
    ]
    assert [row["event_sequence"] for row in rows] == [1, 2, 3]
    assert [row["decision"]["disposition"] for row in rows] == [
        DecisionDisposition.HOLD,
        DecisionDisposition.FILLED,
        DecisionDisposition.REJECTED,
    ]
    assert all(row["paper_only"] is True for row in rows)
    assert all(row["polymarket_write_enabled"] is False for row in rows)

    resumed = PaperTradingSession.resume_existing(
        runner=_runner(),
        output_dir=tmp_path,
        run_id="paper-e2e",
        source_commit="fbd64d9",
        config={"feed": "offline-fixture"},
    )
    assert resumed.current_snapshot == before_resume
    assert resumed.runner.oms.positions()
    resumed.runner.push_alpha_tick(_alpha(100_100, 100_010.0))
    assert resumed.process_snapshot_sync(fill_snapshot) is None
    assert resumed.current_snapshot == before_resume
    assert resumed.runner.decision_count == 0
    assert len(
        (resumed.store.run_dir / SIGNAL_EVENTS_FILE).read_text().splitlines()
    ) == 3
    variant = replace(
        _snapshot(100_100, yes_bid=0.39, yes_ask=0.40, no_bid=0.09, no_ask=0.90),
        yes_ask_size=200.0,
    )
    assert resumed.process_snapshot_sync(variant) is None
    after_signal_replay = resumed.current_snapshot
    assert after_signal_replay.cash == before_resume.cash
    assert after_signal_replay.open_lots == before_resume.open_lots
    assert resumed.runner.last_decision is not None
    assert resumed.runner.last_decision.disposition is DecisionDisposition.NO_ORDER
    settlement = resumed.settle(_settlement())
    final = resumed.current_snapshot
    assert settlement.event_sequence == 5
    assert final.last_event_sequence == 5
    assert final.positions == final.open_lots == ()
    assert final.equity == final.cash
    assert resumed.runner.oms.positions() == ()
    assert final.cash == pytest.approx(resumed.runner.current_bankroll)
    assert final.cash == pytest.approx(resumed.runner.oms.bankroll)
    assert len((resumed.store.run_dir / SETTLEMENT_EVENTS_FILE).read_text().splitlines()) == 1
    assert len((resumed.store.run_dir / LEDGER_EVENTS_FILE).read_text().splitlines()) == 5


def test_resume_rejects_changed_configuration(tmp_path: Path) -> None:
    PaperTradingSession.create_new(
        runner=_runner(fee_bps=0.0),
        output_dir=tmp_path,
        run_id="config-test",
        source_commit="commit-a",
        created_at="2026-09-03T00:00:00+00:00",
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        PaperTradingSession.resume_existing(
            runner=_runner(fee_bps=50.0),
            output_dir=tmp_path,
            run_id="config-test",
            source_commit="commit-a",
        )


@pytest.mark.parametrize(
    (
        "ema_alpha",
        "zscore_min_samples",
        "spot_price",
        "oms_symbol",
        "provider_identity",
    ),
    [
        (1.0, 20, 100_000.0, "BTC", "pricing-feed-v1"),
        (0.2, 1, 100_000.0, "BTC", "pricing-feed-v1"),
        (0.2, 20, 90_000.0, "BTC", "pricing-feed-v1"),
        (0.2, 20, 100_000.0, "ETH", "pricing-feed-v1"),
        (0.2, 20, 100_000.0, "BTC", "pricing-feed-v2"),
    ],
)
def test_resume_rejects_each_strategy_identity_change(
    tmp_path: Path,
    ema_alpha: float,
    zscore_min_samples: int,
    spot_price: float,
    oms_symbol: str,
    provider_identity: str,
) -> None:
    PaperTradingSession.create_new(
        runner=_runner(
            ema_alpha=0.2,
            zscore_min_samples=20,
            spot_price=100_000.0,
            oms_symbol="BTC",
            provider_identity="pricing-feed-v1",
        ),
        output_dir=tmp_path,
        run_id="strategy-identity-test",
        source_commit="commit-a",
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        PaperTradingSession.resume_existing(
            runner=_runner(
                ema_alpha=ema_alpha,
                zscore_min_samples=zscore_min_samples,
                spot_price=spot_price,
                oms_symbol=oms_symbol,
                provider_identity=provider_identity,
            ),
            output_dir=tmp_path,
            run_id="strategy-identity-test",
            source_commit="commit-a",
        )


def test_dynamic_pricing_provider_requires_explicit_identity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="pricing_inputs_provider_identity"):
        PaperTradingSession.create_new(
            runner=_runner(provider_identity=None),
            output_dir=tmp_path,
            run_id="missing-provider-identity",
            source_commit="commit-a",
        )


def test_runner_can_be_owned_by_only_one_paper_session(tmp_path: Path) -> None:
    runner = _runner()
    first = PaperTradingSession.create_new(
        runner=runner,
        output_dir=tmp_path,
        run_id="exclusive-run-a",
        source_commit="commit-a",
    )

    assert first.runner.paper_session_bound is True
    with pytest.raises(ValueError, match="unbound fresh StrategyRunner"):
        PaperTradingSession.create_new(
            runner=runner,
            output_dir=tmp_path,
            run_id="exclusive-run-b",
            source_commit="commit-a",
        )
    assert not (tmp_path / "exclusive-run-b").exists()


@pytest.mark.asyncio
async def test_paper_owned_runner_cannot_start_outside_session(tmp_path: Path) -> None:
    runner = _runner()
    PaperTradingSession.create_new(
        runner=runner,
        output_dir=tmp_path,
        run_id="exclusive-start",
        source_commit="commit-a",
    )

    with pytest.raises(RuntimeError, match="PaperTradingSession"):
        await runner.start()


def test_paper_owned_runner_rejects_direct_sync_processing_before_fill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = PaperTradingSession.create_new(
        runner=_runner(spot_price=200_000.0),
        output_dir=tmp_path,
        run_id="exclusive-sync-processing",
        source_commit="commit-a",
    )

    def fail_append(**_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(session.store, "append_decision", fail_append)
    fill_snapshot = _snapshot(
        100_000,
        yes_bid=0.39,
        yes_ask=0.40,
        no_bid=0.09,
        no_ask=0.90,
    )

    with pytest.raises(RuntimeError, match="PaperTradingSession"):
        session.runner.process_snapshot_sync(fill_snapshot)
    assert session.failed is False
    assert session.runner.decision_count == 0
    assert session.runner.oms.positions() == ()
    assert session.current_snapshot.cash == 1_000.0


@pytest.mark.asyncio
async def test_paper_owned_runner_rejects_direct_async_and_wrong_token(
    tmp_path: Path,
) -> None:
    session = PaperTradingSession.create_new(
        runner=_runner(spot_price=200_000.0),
        output_dir=tmp_path,
        run_id="exclusive-async-processing",
        source_commit="commit-a",
    )
    fill_snapshot = _snapshot(
        100_000,
        yes_bid=0.39,
        yes_ask=0.40,
        no_bid=0.09,
        no_ask=0.90,
    )

    with pytest.raises(RuntimeError, match="PaperTradingSession"):
        await session.runner.process_snapshot(fill_snapshot)
    with pytest.raises(RuntimeError, match="owner token"):
        session.runner._process_paper_snapshot_sync(
            fill_snapshot,
            owner_token=object(),
        )
    assert session.runner.decision_count == 0
    assert session.runner.oms.positions() == ()
    assert session.current_snapshot.cash == 1_000.0


@pytest.mark.asyncio
async def test_session_async_path_uses_owner_token_and_persists_fill(
    tmp_path: Path,
) -> None:
    session = PaperTradingSession.create_new(
        runner=_runner(spot_price=200_000.0),
        output_dir=tmp_path,
        run_id="tokenized-async-session",
        source_commit="commit-a",
    )

    result = await session.process_snapshot(
        _snapshot(
            100_000,
            yes_bid=0.39,
            yes_ask=0.40,
            no_bid=0.09,
            no_ask=0.90,
        )
    )

    assert result is not None and result.status == "FILLED"
    assert session.failed is False
    assert session.current_snapshot.last_event_sequence == 1
    assert len(session.store.load_decision_events()) == 1


def test_snapshot_dedupe_uses_bounded_lru_with_complete_disk_fallback(
    tmp_path: Path,
) -> None:
    session = PaperTradingSession.create_new(
        runner=_runner(),
        output_dir=tmp_path,
        run_id="bounded-snapshot-dedupe",
        source_commit="commit-a",
        snapshot_dedupe_cache_size=2,
    )
    snapshots = [
        _snapshot(
            100_000 + index,
            yes_bid=0.98,
            yes_ask=0.99,
            no_bid=0.0,
            no_ask=0.99,
        )
        for index in range(3)
    ]
    for snapshot in snapshots:
        assert session.process_snapshot_sync(snapshot) is None

    assert session.snapshot_dedupe_cache_entries == 2
    assert session.runner.decision_count == 3
    assert session.process_snapshot_sync(snapshots[0]) is None
    assert session.snapshot_dedupe_cache_entries == 2
    assert session.runner.decision_count == 3


def test_resume_signal_dedupe_is_not_limited_by_oms_cache(tmp_path: Path) -> None:
    first_runner = _runner(
        spot_price=200_000.0,
        zscore_min_samples=1,
        signal_cache_size=1,
    )
    session = PaperTradingSession.create_new(
        runner=first_runner,
        output_dir=tmp_path,
        run_id="durable-signal-dedupe",
        source_commit="commit-a",
    )
    first_snapshot = replace(
        _snapshot(
            100_000,
            yes_bid=0.39,
            yes_ask=0.40,
            no_bid=0.09,
            no_ask=0.90,
        ),
        yes_ask_size=10.0,
    )
    second_snapshot = replace(first_snapshot, timestamp_ms=100_100)
    first_fill = session.process_snapshot_sync(first_snapshot)
    second_fill = session.process_snapshot_sync(second_snapshot)
    assert first_fill is not None and first_fill.status == "FILLED"
    assert second_fill is not None and second_fill.status == "FILLED"
    before_resume = session.current_snapshot

    resumed = PaperTradingSession.resume_existing(
        runner=_runner(
            spot_price=200_000.0,
            zscore_min_samples=1,
            signal_cache_size=1,
        ),
        output_dir=tmp_path,
        run_id="durable-signal-dedupe",
        source_commit="commit-a",
    )
    replay_with_different_liquidity = replace(first_snapshot, yes_ask_size=200.0)

    assert resumed.process_snapshot_sync(replay_with_different_liquidity) is None
    assert resumed.current_snapshot.cash == before_resume.cash
    assert resumed.current_snapshot.open_lots == before_resume.open_lots
    assert resumed.runner.oms_calls == 0
    assert resumed.runner.last_decision is not None
    assert resumed.runner.last_decision.reason_code.value == "duplicate_signal"


def test_all_cash_fill_reserves_fee_and_keeps_every_ledger_consistent(
    tmp_path: Path,
) -> None:
    runner = StrategyRunner(
        ofi_engine=BinanceOFICalculator(zscore_min_samples=1),
        pricing_engine=PolymarketPricingEngine(kelly_fraction=1.0),
        oms=PolymarketOMS(
            max_single_trade_pct=1.0,
            max_position_pct=1.0,
            max_window_exposure_pct=1.0,
            slippage_tolerance=0.0,
        ),
        feed_handler=PolymarketFeedHandler(window_id=WINDOW.window_id, mock=True),
        window=WINDOW,
        initial_bankroll=1_000.0,
        spot_price=200_000.0,
        volatility_annualized=0.10,
        pricing_inputs_provider=lambda timestamp_ms: PricingInputs(
            timestamp_ms=timestamp_ms,
            spot_price=200_000.0,
            oracle_twap_so_far=100_000.0,
            twap_weight=0.0,
            volatility_annualized=0.10,
        ),
        pricing_inputs_provider_identity="all-cash-fixture-v1",
        fee_bps=100.0,
    )
    session = PaperTradingSession.create_new(
        runner=runner,
        output_dir=tmp_path,
        run_id="all-cash-fee-test",
        source_commit="commit-a",
    )

    result = session.process_snapshot_sync(
        replace(
            _snapshot(100_000, yes_bid=0.50, yes_ask=0.50, no_bid=0.0, no_ask=0.99),
            yes_ask_size=10_000.0,
        )
    )

    assert result is not None and result.status == "FILLED"
    assert result.fee_usdc == pytest.approx(result.shares * result.price * 0.01)
    assert session.failed is False
    assert session.current_snapshot.cash == pytest.approx(0.0, abs=1e-9)
    assert session.current_snapshot.cash == pytest.approx(runner.current_bankroll)
    assert runner.current_bankroll == pytest.approx(runner.oms.bankroll)
    assert len(session.current_snapshot.open_lots) == 1
    assert len(runner.oms.positions()) == 1


def test_window_mismatch_is_a_durable_non_mutating_decision(tmp_path: Path) -> None:
    session = PaperTradingSession.create_new(
        runner=_runner(),
        output_dir=tmp_path,
        run_id="mismatch-test",
        source_commit="commit-a",
    )
    foreign = _snapshot(
        100_000,
        yes_bid=0.39,
        yes_ask=0.40,
        no_bid=0.09,
        no_ask=0.90,
    )
    foreign = MarketSnapshot(
        timestamp_ms=foreign.timestamp_ms,
        window_id="foreign-window",
        yes_bid=foreign.yes_bid,
        yes_ask=foreign.yes_ask,
        no_bid=foreign.no_bid,
        no_ask=foreign.no_ask,
        last_traded_price=foreign.last_traded_price,
        yes_bid_size=foreign.yes_bid_size,
        yes_ask_size=foreign.yes_ask_size,
        no_bid_size=foreign.no_bid_size,
        no_ask_size=foreign.no_ask_size,
    )
    assert session.process_snapshot_sync(foreign) is None
    assert session.failed is False
    assert session.current_snapshot.cash == 1_000.0
    assert session.current_snapshot.last_event_sequence == 1


def test_storage_failure_permanently_fails_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = PaperTradingSession.create_new(
        runner=_runner(),
        output_dir=tmp_path,
        run_id="failure-test",
        source_commit="commit-a",
    )

    def fail_append(**_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(session.store, "append_decision", fail_append)
    with pytest.raises(PaperSessionFailedError, match="disk full"):
        session.process_snapshot_sync(
            _snapshot(100_000, yes_bid=0.98, yes_ask=0.99, no_bid=0.0, no_ask=0.99)
        )
    assert session.failed is True
    with pytest.raises(PaperSessionFailedError, match="disk full"):
        session.process_snapshot_sync(
            _snapshot(100_001, yes_bid=0.39, yes_ask=0.40, no_bid=0.09, no_ask=0.90)
        )


def test_runner_exception_checks_consistency_and_fails_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = PaperTradingSession.create_new(
        runner=_runner(),
        output_dir=tmp_path,
        run_id="runner-exception-test",
        source_commit="commit-a",
    )

    def corrupt_then_raise(self: PolymarketOMS, *_args: object, **_kwargs: object) -> None:
        self.bankroll = 0.0
        raise RuntimeError("simulated OMS failure")

    monkeypatch.setattr(PolymarketOMS, "process_signal", corrupt_then_raise)
    with pytest.raises(RuntimeError, match="simulated OMS failure"):
        session.process_snapshot_sync(
            _snapshot(100_000, yes_bid=0.39, yes_ask=0.40, no_bid=0.09, no_ask=0.90)
        )
    assert session.failed is True
    assert session.failure_reason is not None
    assert "inconsistent" in session.failure_reason
    assert session.current_snapshot.positions == ()
