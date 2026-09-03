"""CLOB event-driven backtest engine, loader, and metrics."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from bigan.backtest.data_loader import (
    generate_synthetic_clob,
    load_clob_snapshots,
    snapshots_from_table,
    write_clob_snapshots,
)
from bigan.backtest.engine import BacktestEngine, StrategyBacktestParams
from bigan.backtest.metrics import ClosedTrade, compute_backtest_metrics
from bigan.data.polymarket_clob import MarketSnapshot, PolymarketFeedHandler
from bigan.execution.polymarket_oms import PolymarketOMS
from bigan.features.binance_ofi import BinanceOFICalculator
from bigan.pipeline.strategy_runner import PricingInputs, StrategyRunner
from bigan.strategies.polymarket_pricing import MarketWindow, PolymarketPricingEngine

WINDOW_ID = "btc-updown-15m-bt"


def _window() -> MarketWindow:
    return MarketWindow(
        window_id=WINDOW_ID,
        symbol="BTC",
        strike_price=100_000.0,
        start_ts_ms=0,
        end_ts_ms=900_000,
        window_type="15m",
    )


def _params(**overrides: object) -> StrategyBacktestParams:
    values: dict[str, object] = {
        "ofi_zscore_min_samples": 5,
        "ofi_ema_alpha": 1.0,
        "initial_bankroll": 1_000.0,
        "spot_price": 100_000.0,
        "fee_bps": 0.0,
        "execution_mode": "market",
    }
    values.update(overrides)
    return StrategyBacktestParams(**values)  # type: ignore[arg-type]


def _engine(**overrides: object) -> BacktestEngine:
    return BacktestEngine(window=_window(), params=_params(**overrides))


def test_metrics_hand_calculated_equity_and_trades() -> None:
    trades = (
        ClosedTrade(
            window_id=WINDOW_ID,
            side="YES",
            entry_ts_ms=0,
            exit_ts_ms=1_000,
            shares=10.0,
            entry_price=0.40,
            exit_price=1.0,
            pnl=10.0,
            fee_usdc=0.5,
        ),
        ClosedTrade(
            window_id=WINDOW_ID,
            side="YES",
            entry_ts_ms=1_000,
            exit_ts_ms=2_000,
            shares=10.0,
            entry_price=0.50,
            exit_price=0.0,
            pnl=-5.0,
            fee_usdc=0.5,
        ),
        ClosedTrade(
            window_id=WINDOW_ID,
            side="NO",
            entry_ts_ms=2_000,
            exit_ts_ms=3_000,
            shares=5.0,
            entry_price=0.40,
            exit_price=1.0,
            pnl=15.0,
            fee_usdc=0.25,
        ),
    )
    report = compute_backtest_metrics(
        equity_ts_ms=(0, 1_000, 2_000, 3_000),
        equity=(100.0, 110.0, 105.0, 120.0),
        trades=trades,
        initial_bankroll=100.0,
        commission_paid=1.25,
    )
    assert report.total_return == pytest.approx(0.20)
    assert report.max_drawdown == pytest.approx((110.0 - 105.0) / 110.0)
    assert report.win_rate == pytest.approx(2.0 / 3.0)
    assert report.profit_factor == pytest.approx(5.0)
    assert report.total_trades == 3
    assert report.avg_pnl_per_trade == pytest.approx(20.0 / 3.0)
    assert report.avg_trade_duration_ms == pytest.approx(1_000.0)
    assert report.commission_paid == pytest.approx(1.25)
    assert report.sharpe_ratio != 0.0


def test_clob_loader_roundtrip_csv_and_parquet(tmp_path: Path) -> None:
    snapshots = generate_synthetic_clob(n_ticks=12, window_id=WINDOW_ID, seed=7)
    spots = tuple(100_000.0 + i for i in range(len(snapshots)))
    csv_path = tmp_path / "clob.csv"
    parquet_path = tmp_path / "clob.parquet"
    write_clob_snapshots(csv_path, snapshots, spot_prices=spots)
    write_clob_snapshots(parquet_path, snapshots, spot_prices=spots)
    csv_loaded = load_clob_snapshots(csv_path)
    pq_loaded = load_clob_snapshots(parquet_path)
    assert csv_loaded.snapshots == snapshots
    assert pq_loaded.snapshots == snapshots
    assert csv_loaded.spot_prices == pytest.approx(spots)
    assert pq_loaded.spot_prices == pytest.approx(spots)
    assert csv_loaded.dropped_stale == 0


def test_clob_loader_requires_execution_ask_sizes() -> None:
    table = pa.table(
        {
            "timestamp_ms": [100],
            "window_id": [WINDOW_ID],
            "yes_bid": [0.38],
            "yes_ask": [0.42],
            "no_bid": [0.38],
            "no_ask": [0.42],
        }
    )

    with pytest.raises(ValueError, match="yes_ask_size"):
        snapshots_from_table(table)


def test_minimum_documented_clob_schema_can_fill_orders() -> None:
    table = pa.table(
        {
            "timestamp_ms": [100_000, 100_200, 100_400],
            "window_id": [WINDOW_ID] * 3,
            "yes_bid": [0.39] * 3,
            "yes_ask": [0.40] * 3,
            "no_bid": [0.09] * 3,
            "no_ask": [0.90] * 3,
            "yes_ask_size": [100.0] * 3,
            "no_ask_size": [100.0] * 3,
        }
    )

    loaded = snapshots_from_table(table)
    result = _engine().run(loaded.snapshots, settlement={WINDOW_ID: 1.0})

    assert loaded.dropped_stale == 0
    assert all(row.yes_ask_size == 100.0 for row in loaded.snapshots)
    assert any(row.order.status == "FILLED" for row in result.fills)


def test_clob_loader_drops_invalid_spot_rows_atomically() -> None:
    table = pa.table(
        {
            "timestamp_ms": [100, 200, 300, 400, 500],
            "window_id": [WINDOW_ID] * 5,
            "yes_bid": [0.38] * 5,
            "yes_ask": [0.42] * 5,
            "no_bid": [0.38] * 5,
            "no_ask": [0.42] * 5,
            "yes_ask_size": [100.0] * 5,
            "no_ask_size": [100.0] * 5,
            "spot_price": [100_000.0, None, 0.0, float("nan"), 100_004.0],
        }
    )

    loaded = snapshots_from_table(table)

    assert [row.timestamp_ms for row in loaded.snapshots] == [100, 500]
    assert loaded.spot_prices == pytest.approx((100_000.0, 100_004.0))
    assert len(loaded.snapshots) == len(loaded.spot_prices or ())
    assert loaded.dropped_stale == 3


def test_clob_loader_rejects_null_and_non_string_window_ids() -> None:
    table = pa.table(
        {
            "timestamp_ms": [100, 200, 300, 400],
            "window_id": [None, "", "   ", f"  {WINDOW_ID}  "],
            "yes_bid": [0.38] * 4,
            "yes_ask": [0.42] * 4,
            "no_bid": [0.38] * 4,
            "no_ask": [0.42] * 4,
            "yes_ask_size": [100.0] * 4,
            "no_ask_size": [100.0] * 4,
        }
    )

    loaded = snapshots_from_table(table)

    assert [row.timestamp_ms for row in loaded.snapshots] == [400]
    assert loaded.snapshots[0].window_id == WINDOW_ID
    assert loaded.dropped_stale == 3

    numeric_ids = table.set_column(
        table.schema.get_field_index("window_id"),
        "window_id",
        pa.array([1, 2, 3, 4]),
    )
    numeric_loaded = snapshots_from_table(numeric_ids)
    assert numeric_loaded.snapshots == ()
    assert numeric_loaded.dropped_stale == 4


@pytest.mark.parametrize(
    ("column", "invalid_value", "valid_value"),
    [
        ("yes_ask", 1.20, 0.42),
        ("last_traded_price", -0.50, 0.40),
        ("last_traded_price", float("nan"), 0.40),
    ],
)
def test_clob_loader_drops_invalid_binary_contract_prices(
    column: str,
    invalid_value: float,
    valid_value: float,
) -> None:
    payload: dict[str, list[object]] = {
        "timestamp_ms": [100, 200],
        "window_id": [WINDOW_ID, WINDOW_ID],
        "yes_bid": [0.38, 0.38],
        "yes_ask": [0.42, 0.42],
        "no_bid": [0.38, 0.38],
        "no_ask": [0.42, 0.42],
        "yes_ask_size": [100.0, 100.0],
        "no_ask_size": [100.0, 100.0],
    }
    payload[column] = [invalid_value, valid_value]

    loaded = snapshots_from_table(pa.table(payload))

    assert [row.timestamp_ms for row in loaded.snapshots] == [200]
    assert loaded.dropped_stale == 1


def test_engine_fills_match_strategy_runner_and_oms_cash() -> None:
    snapshots = generate_synthetic_clob(n_ticks=40, window_id=WINDOW_ID, seed=1)
    engine = _engine()
    result = engine.run(snapshots, settlement={WINDOW_ID: 1.0})
    filled = [row for row in result.fills if row.order.status == "FILLED"]
    assert filled
    assert filled[-1].order.side == "YES"
    assert filled[-1].slippage >= 0.0

    runner = StrategyRunner(
        ofi_engine=BinanceOFICalculator(zscore_min_samples=5, ema_alpha=1.0),
        pricing_engine=PolymarketPricingEngine(),
        oms=PolymarketOMS(max_spread_allowed=0.08, slippage_tolerance=0.01),
        feed_handler=PolymarketFeedHandler(
            window_id=WINDOW_ID,
            yes_token_id="yes",
            no_token_id="no",
            mock=True,
        ),
        window=_window(),
        initial_bankroll=1_000.0,
        spot_price=100_000.0,
        pricing_inputs_provider=lambda timestamp_ms: PricingInputs(
            timestamp_ms=timestamp_ms,
            spot_price=100_000.0,
            oracle_twap_so_far=100_000.0,
            twap_weight=0.0,
            volatility_annualized=0.60,
        ),
    )
    for snap in snapshots:
        runner.process_snapshot_sync(snap)
    runner_filled = [row for row in runner.execution_history if row.status == "FILLED"]
    assert len(filled) == len(runner_filled)
    for engine_fill, oms_fill in zip(filled, runner_filled, strict=True):
        assert engine_fill.order.side == oms_fill.side
        assert engine_fill.order.shares == pytest.approx(oms_fill.shares)
        assert engine_fill.order.price == pytest.approx(oms_fill.price)
    assert filled[-1].cash_after == pytest.approx(runner.current_bankroll)
    assert result.final_cash > filled[-1].cash_after
    assert result.metrics.total_trades == len(filled)
    assert result.metrics.total_return > 0.0
    assert result.metrics.commission_paid == pytest.approx(0.0)


def test_engine_fees_reduce_cash_vs_zero_fee() -> None:
    snapshots = generate_synthetic_clob(n_ticks=40, window_id=WINDOW_ID, seed=2)
    free = _engine(fee_bps=0.0).run(snapshots, settlement={WINDOW_ID: 1.0})
    taxed = _engine(fee_bps=100.0).run(snapshots, settlement={WINDOW_ID: 1.0})
    assert taxed.metrics.commission_paid > 0.0
    assert taxed.final_cash < free.final_cash
    filled = [row for row in taxed.fills if row.order.status == "FILLED"]
    assert filled
    assert filled[0].fee_usdc == pytest.approx(filled[0].order.shares * filled[0].order.price * 0.01)


def test_zscore_gate_blocks_oms() -> None:
    snapshots = generate_synthetic_clob(n_ticks=40, window_id=WINDOW_ID, seed=3)
    result = _engine(min_abs_z_ofi=100.0).run(snapshots, settlement={WINDOW_ID: 1.0})
    assert result.oms_calls == 0
    assert result.fills == ()
    assert result.final_cash == pytest.approx(1_000.0)
    assert result.metrics.total_trades == 0


def test_limit_order_rests_then_fills_when_ask_crosses() -> None:
    snaps = [
        MarketSnapshot(
            timestamp_ms=100_000 + i * 200,
            window_id=WINDOW_ID,
            yes_bid=0.38,
            yes_ask=0.42,
            no_bid=0.48,
            no_ask=0.90,
            last_traded_price=0.38,
            yes_bid_size=10_000.0,
            yes_ask_size=10_000.0,
            no_bid_size=10_000.0,
            no_ask_size=10_000.0,
        )
        for i in range(6)
    ]
    snaps.append(
        MarketSnapshot(
            timestamp_ms=101_400,
            window_id=WINDOW_ID,
            yes_bid=0.36,
            yes_ask=0.37,
            no_bid=0.48,
            no_ask=0.90,
            last_traded_price=0.37,
            yes_bid_size=10_000.0,
            yes_ask_size=10_000.0,
            no_bid_size=10_000.0,
            no_ask_size=10_000.0,
        )
    )
    result = _engine(execution_mode="limit", ofi_zscore_min_samples=1).run(
        snaps,
        settlement={WINDOW_ID: 1.0},
    )
    filled = [row for row in result.fills if row.order.status == "FILLED"]
    assert filled
    assert filled[0].timestamp_ms == 101_400
    assert filled[0].order.side == "YES"
    assert filled[0].ask_price == pytest.approx(0.37)
    assert filled[0].order.price == pytest.approx(0.37 * 1.01)
    assert filled[0].order.price <= 0.38


def test_marketable_limit_order_fills_on_submission_snapshot() -> None:
    snapshot = MarketSnapshot(
        timestamp_ms=100_000,
        window_id=WINDOW_ID,
        yes_bid=0.40,
        yes_ask=0.40,
        no_bid=0.09,
        no_ask=0.90,
        last_traded_price=0.40,
        yes_bid_size=10_000.0,
        yes_ask_size=10_000.0,
        no_bid_size=10_000.0,
        no_ask_size=10_000.0,
    )

    result = _engine(execution_mode="limit").run(
        (snapshot,),
        spot_prices=(101_000.0,),
        settlement={WINDOW_ID: 1.0},
    )

    filled = [row for row in result.fills if row.order.status == "FILLED"]
    assert len(filled) == 1
    assert filled[0].timestamp_ms == snapshot.timestamp_ms
    assert filled[0].order.price == pytest.approx(0.40)
    assert filled[0].order.shares == pytest.approx(50.0 / 0.40)


def test_resting_limit_fills_before_tail_cutoff_cancels_new_signal() -> None:
    window = MarketWindow(
        window_id=WINDOW_ID,
        symbol="BTC",
        strike_price=100_000.0,
        start_ts_ms=0,
        end_ts_ms=145_000,
        window_type="15m",
    )
    snapshots = (
        MarketSnapshot(
            timestamp_ms=100_000,
            window_id=WINDOW_ID,
            yes_bid=0.38,
            yes_ask=0.42,
            no_bid=0.09,
            no_ask=0.90,
            last_traded_price=0.38,
            yes_bid_size=10_000.0,
            yes_ask_size=10_000.0,
            no_bid_size=10_000.0,
            no_ask_size=10_000.0,
        ),
        MarketSnapshot(
            timestamp_ms=115_000,
            window_id=WINDOW_ID,
            yes_bid=0.36,
            yes_ask=0.37,
            no_bid=0.09,
            no_ask=0.90,
            last_traded_price=0.37,
            yes_bid_size=10_000.0,
            yes_ask_size=10_000.0,
            no_bid_size=10_000.0,
            no_ask_size=10_000.0,
        ),
    )
    result = BacktestEngine(
        window=window,
        params=_params(execution_mode="limit", ofi_zscore_min_samples=1),
    ).run(snapshots, settlement={WINDOW_ID: 1.0})
    filled = [row for row in result.fills if row.order.status == "FILLED"]
    assert len(filled) == 1
    assert filled[0].timestamp_ms == 115_000
    assert filled[0].order.price <= 0.38


def test_limit_direction_reversal_replaces_old_resting_order() -> None:
    snapshots = (
        MarketSnapshot(
            timestamp_ms=100_000,
            window_id=WINDOW_ID,
            yes_bid=0.38,
            yes_ask=0.42,
            no_bid=0.38,
            no_ask=0.42,
            last_traded_price=0.40,
            yes_ask_size=10_000.0,
            no_ask_size=10_000.0,
        ),
        MarketSnapshot(
            timestamp_ms=100_200,
            window_id=WINDOW_ID,
            yes_bid=0.38,
            yes_ask=0.42,
            no_bid=0.38,
            no_ask=0.42,
            last_traded_price=0.40,
            yes_ask_size=10_000.0,
            no_ask_size=10_000.0,
        ),
        MarketSnapshot(
            timestamp_ms=100_400,
            window_id=WINDOW_ID,
            yes_bid=0.20,
            yes_ask=0.37,
            no_bid=0.36,
            no_ask=0.37,
            last_traded_price=0.37,
            yes_ask_size=10_000.0,
            no_ask_size=10_000.0,
        ),
    )
    result = _engine(execution_mode="limit").run(
        snapshots,
        spot_prices=(101_000.0, 99_000.0, 99_000.0),
        settlement={WINDOW_ID: 0.0},
    )

    filled = [row for row in result.fills if row.order.status == "FILLED"]
    assert len(filled) == 1
    assert filled[0].timestamp_ms == 100_400
    assert filled[0].order.side == "NO"


def test_limit_order_shares_are_fixed_before_price_improvement() -> None:
    def filled_shares(crossed_ask: float) -> float:
        snapshots = (
            MarketSnapshot(
                timestamp_ms=100_000,
                window_id=WINDOW_ID,
                yes_bid=0.38,
                yes_ask=0.42,
                no_bid=0.38,
                no_ask=0.42,
                last_traded_price=0.40,
                yes_ask_size=10_000.0,
                no_ask_size=10_000.0,
            ),
            MarketSnapshot(
                timestamp_ms=100_200,
                window_id=WINDOW_ID,
                yes_bid=0.20 if crossed_ask > 0.20 else 0.19,
                yes_ask=crossed_ask,
                no_bid=0.38,
                no_ask=0.42,
                last_traded_price=crossed_ask,
                yes_ask_size=10_000.0,
                no_ask_size=10_000.0,
            ),
        )
        result = _engine(execution_mode="limit").run(
            snapshots,
            spot_prices=(101_000.0, 101_000.0),
            settlement={WINDOW_ID: 1.0},
        )
        filled = [row for row in result.fills if row.order.status == "FILLED"]
        assert len(filled) == 1
        return filled[0].order.shares

    at_037 = filled_shares(0.37)
    at_020 = filled_shares(0.20)
    assert at_037 == pytest.approx(50.0 / 0.38)
    assert at_020 == pytest.approx(at_037)


def test_multi_window_replay_uses_each_window_metadata() -> None:
    first_window = MarketWindow(
        window_id="window-1",
        symbol="BTC",
        strike_price=100_000.0,
        start_ts_ms=0,
        end_ts_ms=300_000,
        window_type="5m",
    )
    second_window = MarketWindow(
        window_id="window-2",
        symbol="BTC",
        strike_price=100_000.0,
        start_ts_ms=300_000,
        end_ts_ms=1_200_000,
        window_type="15m",
    )
    snapshots = tuple(
        MarketSnapshot(
            timestamp_ms=timestamp_ms,
            window_id=window_id,
            yes_bid=0.39,
            yes_ask=0.40,
            no_bid=0.09,
            no_ask=0.90,
            last_traded_price=0.40,
            yes_bid_size=10_000.0,
            yes_ask_size=10_000.0,
            no_bid_size=10_000.0,
            no_ask_size=10_000.0,
        )
        for timestamp_ms, window_id in ((100_000, "window-1"), (400_000, "window-2"))
    )
    engine = BacktestEngine(
        window=first_window,
        windows={second_window.window_id: second_window},
        params=_params(),
    )
    result = engine.run(
        snapshots,
        settlement={"window-1": 1.0, "window-2": 1.0},
    )
    filled_windows = [row.window_id for row in result.fills if row.order.status == "FILLED"]
    assert filled_windows == ["window-1", "window-2"]


def test_multi_window_replay_rejects_missing_metadata() -> None:
    snapshot = MarketSnapshot(
        timestamp_ms=400_000,
        window_id="window-2",
        yes_bid=0.39,
        yes_ask=0.40,
        no_bid=0.09,
        no_ask=0.90,
        last_traded_price=0.40,
        yes_bid_size=10_000.0,
        yes_ask_size=10_000.0,
        no_bid_size=10_000.0,
        no_ask_size=10_000.0,
    )
    with pytest.raises(ValueError, match="missing MarketWindow metadata"):
        _engine().run((snapshot,))


@pytest.mark.parametrize("payout", [2.0, -0.5, float("nan"), float("inf")])
def test_backtest_rejects_invalid_settlement_payouts_before_replay(payout: float) -> None:
    snapshot = MarketSnapshot(
        timestamp_ms=100_000,
        window_id=WINDOW_ID,
        yes_bid=0.39,
        yes_ask=0.40,
        no_bid=0.09,
        no_ask=0.90,
        last_traded_price=0.40,
        yes_ask_size=100.0,
        no_ask_size=100.0,
    )

    with pytest.raises(ValueError, match="settlement payout"):
        _engine().run((snapshot,), settlement={WINDOW_ID: payout})
