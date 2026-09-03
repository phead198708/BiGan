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
    snapshots = generate_synthetic_clob(n_ticks=20, window_id=WINDOW_ID, step_ms=1_000)
    train, test = split_snapshots_by_time(snapshots, train_ratio=0.70)
    assert train
    assert test
    assert train[-1].timestamp_ms <= test[0].timestamp_ms
    assert train[0].timestamp_ms == snapshots[0].timestamp_ms
    assert test[-1].timestamp_ms == snapshots[-1].timestamp_ms
    folds = time_series_folds(snapshots, n_splits=2)
    assert len(folds) == 2
    for is_tape, oos_tape in folds:
        assert is_tape[-1].timestamp_ms <= oos_tape[0].timestamp_ms


def test_grid_search_selects_tradeable_spread_in_process() -> None:
    snapshots = generate_synthetic_clob(n_ticks=60, window_id=WINDOW_ID, seed=11)
    report = run_grid_search(
        snapshots,
        {"max_spread_allowed": (0.01, 0.20)},
        window=_window(),
        base=_base(),
        train_ratio=0.65,
        max_workers=1,
        score="total_return",
        settlement={WINDOW_ID: 1.0},
    )
    assert report.best_params == {"max_spread_allowed": 0.20}
    scores = {trial.params["max_spread_allowed"]: trial.out_of_sample_score for trial in report.trials}
    assert scores[0.20] > scores[0.01]


def test_grid_search_parallel_matches_sequential_winner() -> None:
    snapshots = generate_synthetic_clob(n_ticks=48, window_id=WINDOW_ID, seed=13)
    grid = {
        "max_spread_allowed": (0.01, 0.20),
        "min_abs_z_ofi": (0.0, 100.0),
    }
    kwargs = {
        "window": _window(),
        "base": _base(),
        "train_ratio": 0.65,
        "score": "total_return",
        "settlement": {WINDOW_ID: 1.0},
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
    snapshots = generate_synthetic_clob(
        n_ticks=60,
        window_id=WINDOW_ID,
        yes_ask=0.49,
        start_yes_bid=0.47,
        lift=0.0,
        seed=17,
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
        window=_window(),
        base=base,
        train_ratio=0.65,
        max_workers=2,
        score="total_return",
        settlement={WINDOW_ID: 1.0},
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
