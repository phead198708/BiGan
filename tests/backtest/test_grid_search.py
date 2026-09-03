"""Parallel CLOB parameter grid search with time-series splits."""

from __future__ import annotations

import pytest

from bigan.backtest.data_loader import generate_synthetic_clob
from bigan.backtest.engine import StrategyBacktestParams
from bigan.backtest.grid_search import (
    expand_param_grid,
    run_grid_search,
    split_snapshots_by_time,
    time_series_folds,
)
from bigan.data.polymarket_clob import MarketSnapshot
from bigan.features.binance_ofi import TopOfBook
from bigan.strategies.polymarket_pricing import MarketWindow

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


def _base() -> StrategyBacktestParams:
    return StrategyBacktestParams(
        ofi_zscore_min_samples=5,
        ofi_ema_alpha=1.0,
        initial_bankroll=1_000.0,
        spot_price=100_000.0,
        fee_bps=0.0,
    )


def _multi_window_tape(
    *,
    n_windows: int = 4,
    n_ticks: int = 15,
    yes_ask: float = 0.40,
    start_yes_bid: float = 0.30,
    lift: float = 0.002,
) -> tuple[tuple[MarketWindow, ...], tuple[MarketSnapshot, ...]]:
    windows = tuple(
        MarketWindow(
            window_id=f"{WINDOW_ID}-{index}",
            symbol="BTC",
            strike_price=100_000.0,
            start_ts_ms=index * 1_000_000,
            end_ts_ms=index * 1_000_000 + 900_000,
            window_type="15m",
        )
        for index in range(n_windows)
    )
    snapshots = tuple(
        snapshot
        for index, window in enumerate(windows)
        for snapshot in generate_synthetic_clob(
            n_ticks=n_ticks,
            window_id=window.window_id,
            start_ts_ms=window.start_ts_ms + 100_000,
            yes_ask=yes_ask,
            start_yes_bid=start_yes_bid,
            lift=lift,
            seed=index,
        )
    )
    return windows, snapshots


def test_expand_param_grid_is_cartesian() -> None:
    combos = expand_param_grid(
        {
            "max_spread_allowed": (0.01, 0.20),
            "volatility_annualized": (0.4, 0.6),
        }
    )
    assert len(combos) == 4
    assert combos[0] == {"max_spread_allowed": 0.01, "volatility_annualized": 0.4}
    assert combos[-1] == {"max_spread_allowed": 0.20, "volatility_annualized": 0.6}


def test_expand_param_grid_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="unknown grid"):
        expand_param_grid({"not_a_param": (1,)})


def test_time_split_is_causal() -> None:
    one_window = generate_synthetic_clob(n_ticks=20, window_id=WINDOW_ID, step_ms=1_000)
    train, test = split_snapshots_by_time(one_window, train_ratio=0.70)
    assert train
    assert test
    assert train[-1].timestamp_ms <= test[0].timestamp_ms
    assert train[0].timestamp_ms == one_window[0].timestamp_ms
    assert test[-1].timestamp_ms == one_window[-1].timestamp_ms
    _, snapshots = _multi_window_tape(n_ticks=5)
    folds = time_series_folds(snapshots, n_splits=2)
    assert len(folds) == 2
    for is_tape, oos_tape in folds:
        assert is_tape[-1].timestamp_ms <= oos_tape[0].timestamp_ms


def test_time_series_folds_reject_single_window_outcome_leakage() -> None:
    snapshots = generate_synthetic_clob(n_ticks=20, window_id=WINDOW_ID)
    with pytest.raises(ValueError, match="not enough complete windows"):
        time_series_folds(snapshots)


def test_grid_search_selects_tradeable_spread_in_process() -> None:
    windows, snapshots = _multi_window_tape()
    report = run_grid_search(
        snapshots,
        {"max_spread_allowed": (0.01, 0.20)},
        window=windows[0],
        windows={row.window_id: row for row in windows[1:]},
        base=_base(),
        train_ratio=0.65,
        max_workers=1,
        score="total_return",
        settlement={row.window_id: 1.0 for row in windows},
    )
    assert report.best_params == {"max_spread_allowed": 0.20}
    scores = {trial.params["max_spread_allowed"]: trial.out_of_sample_score for trial in report.trials}
    assert scores[0.20] > scores[0.01]


def test_grid_search_parallel_matches_sequential_winner() -> None:
    windows, snapshots = _multi_window_tape(n_ticks=12)
    grid = {
        "max_spread_allowed": (0.01, 0.20),
        "min_abs_z_ofi": (0.0, 100.0),
    }
    kwargs = {
        "window": windows[0],
        "windows": {row.window_id: row for row in windows[1:]},
        "base": _base(),
        "train_ratio": 0.65,
        "score": "total_return",
        "settlement": {row.window_id: 1.0 for row in windows},
    }
    sequential = run_grid_search(snapshots, grid, max_workers=1, **kwargs)
    parallel = run_grid_search(snapshots, grid, max_workers=2, **kwargs)
    assert sequential.best_params == parallel.best_params
    assert sequential.best_params["max_spread_allowed"] == 0.20
    assert sequential.best_params["min_abs_z_ofi"] == 0.0
    assert len(parallel.trials) == 4
    assert sequential.best_trial.out_of_sample_score == pytest.approx(
        parallel.best_trial.out_of_sample_score
    )


def test_grid_search_splits_and_uses_dynamic_spot_and_alpha_inputs() -> None:
    windows, snapshots = _multi_window_tape(
        yes_ask=0.49,
        start_yes_bid=0.47,
        lift=0.0,
    )
    spots = tuple(100_000.0 for _ in snapshots)
    alpha_books = tuple(
        TopOfBook(
            ts_ms=snapshot.timestamp_ms,
            bid_price=100_000.0,
            bid_qty=10.0 + (i % 2),
            ask_price=100_001.0,
            ask_qty=10.0,
        )
        for i, snapshot in enumerate(snapshots)
    )
    base = StrategyBacktestParams(
        ofi_zscore_min_samples=2,
        ofi_ema_alpha=1.0,
        initial_bankroll=1_000.0,
        spot_price=90_000.0,
        fee_bps=0.0,
    )
    report = run_grid_search(
        snapshots,
        {"ofi_gamma": (0.0, 100.0)},
        window=windows[0],
        windows={row.window_id: row for row in windows[1:]},
        base=base,
        train_ratio=0.65,
        max_workers=2,
        score="total_return",
        settlement={row.window_id: 1.0 for row in windows},
        spot_prices=spots,
        alpha_books=alpha_books,
    )
    scores = {trial.params["ofi_gamma"]: trial.out_of_sample_score for trial in report.trials}
    assert scores[100.0] > scores[0.0]


@pytest.mark.parametrize("field", ["spot_prices", "alpha_books"])
def test_grid_search_rejects_misaligned_auxiliary_inputs(field: str) -> None:
    snapshots = generate_synthetic_clob(n_ticks=10, window_id=WINDOW_ID)
    kwargs: dict[str, object] = {field: (1.0,)}
    with pytest.raises(ValueError, match=f"{field} must match snapshots length"):
        run_grid_search(
            snapshots,
            {"ofi_gamma": (0.0,)},
            window=_window(),
            max_workers=1,
            **kwargs,  # type: ignore[arg-type]
        )


def test_multi_window_grid_search_keeps_complete_windows_in_each_fold() -> None:
    windows = tuple(
        MarketWindow(
            window_id=f"window-{index}",
            symbol="BTC",
            strike_price=100_000.0,
            start_ts_ms=index * 1_000_000,
            end_ts_ms=(index + 1) * 1_000_000,
            window_type="15m",
        )
        for index in range(4)
    )
    snapshots = tuple(
        MarketSnapshot(
            timestamp_ms=window.start_ts_ms + offset,
            window_id=window.window_id,
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
        for window in windows
        for offset in (100_000, 200_000, 300_000)
    )
    folds = time_series_folds(snapshots, train_ratio=0.50)
    assert len(folds) == 1
    train, test = folds[0]
    train_windows = {row.window_id for row in train}
    test_windows = {row.window_id for row in test}
    assert train_windows
    assert test_windows
    assert train_windows.isdisjoint(test_windows)

    spots = tuple(100_000.0 for _ in snapshots)
    alpha_books = tuple(
        TopOfBook(
            ts_ms=snapshot.timestamp_ms,
            bid_price=100_000.0,
            bid_qty=10.0 + (index % 2),
            ask_price=100_001.0,
            ask_qty=10.0,
        )
        for index, snapshot in enumerate(snapshots)
    )
    report = run_grid_search(
        snapshots,
        {"ofi_gamma": (0.0,)},
        window=windows[0],
        windows={window.window_id: window for window in windows[1:]},
        base=_base(),
        train_ratio=0.50,
        max_workers=1,
        settlement={window.window_id: 1.0 for window in windows},
        spot_prices=spots,
        alpha_books=alpha_books,
    )
    assert len(report.trials) == 1
