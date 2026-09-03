"""Parallel parameter grid search with time-series cross-validation."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, fields, replace
from itertools import product
from multiprocessing import get_context

from bigan.data.polymarket_clob import MarketSnapshot
from bigan.features.binance_ofi import TopOfBook
from bigan.strategies.polymarket_pricing import MarketWindow

from .engine import BacktestEngine, StrategyBacktestParams
from .metrics import StrategyBacktestMetrics

_PARAM_NAMES = {item.name for item in fields(StrategyBacktestParams)}
_SCORE_NAMES = {item.name for item in fields(StrategyBacktestMetrics)}


@dataclass(frozen=True, slots=True)
class GridSearchTrial:
    """One parameter combination scored on in-sample and out-of-sample folds."""

    params: dict[str, object]
    in_sample_score: float
    out_of_sample_score: float
    in_sample: tuple[StrategyBacktestMetrics, ...]
    out_of_sample: tuple[StrategyBacktestMetrics, ...]


@dataclass(frozen=True, slots=True)
class GridSearchReport:
    """Ranked grid-search output. ``best_params`` maximizes the OOS score."""

    score: str
    best_params: dict[str, object]
    best_trial: GridSearchTrial
    trials: tuple[GridSearchTrial, ...]


@dataclass(frozen=True, slots=True)
class _GridFold:
    train: tuple[MarketSnapshot, ...]
    test: tuple[MarketSnapshot, ...]
    train_spots: tuple[float, ...] | None
    test_spots: tuple[float, ...] | None
    train_alpha: tuple[TopOfBook, ...] | None
    test_alpha: tuple[TopOfBook, ...] | None


@dataclass(frozen=True, slots=True)
class _GridJob:
    updates: dict[str, object]
    base: StrategyBacktestParams
    window: MarketWindow
    windows: dict[str, MarketWindow]
    folds: tuple[_GridFold, ...]
    settlement: dict[str, float]
    score: str


def expand_param_grid(grid: Mapping[str, Sequence[object]]) -> tuple[dict[str, object], ...]:
    """Cartesian product of a parameter grid, in key-then-value order."""

    if not grid:
        return ({},)
    unknown = set(grid) - _PARAM_NAMES
    if unknown:
        raise ValueError(f"unknown grid parameters: {sorted(unknown)}")
    keys = tuple(grid)
    value_rows = [tuple(grid[key]) for key in keys]
    if any(len(row) == 0 for row in value_rows):
        raise ValueError("grid values must be non-empty sequences")
    combos: list[dict[str, object]] = []
    for combo in product(*value_rows):
        combos.append(dict(zip(keys, combo, strict=True)))
    return tuple(combos)


def apply_param_updates(
    base: StrategyBacktestParams,
    updates: Mapping[str, object],
) -> StrategyBacktestParams:
    unknown = set(updates) - _PARAM_NAMES
    if unknown:
        raise ValueError(f"unknown parameters: {sorted(unknown)}")
    return replace(base, **dict(updates))  # type: ignore[arg-type]


def split_snapshots_by_time(
    snapshots: Sequence[MarketSnapshot],
    *,
    train_ratio: float = 0.70,
) -> tuple[tuple[MarketSnapshot, ...], tuple[MarketSnapshot, ...]]:
    """Split a tape by elapsed time, not by row count."""

    if not 0.0 < float(train_ratio) < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    tape = tuple(snapshots)
    if len(tape) < 2:
        return tape, ()
    start = tape[0].timestamp_ms
    end = tape[-1].timestamp_ms
    span = end - start
    if span <= 0:
        cut = max(1, int(len(tape) * train_ratio))
        return tape[:cut], tape[cut:]
    cut_ts = start + int(span * train_ratio)
    cut = 1
    for i, row in enumerate(tape):
        if row.timestamp_ms <= cut_ts:
            cut = i + 1
        else:
            break
    cut = min(max(cut, 1), len(tape) - 1)
    return tape[:cut], tape[cut:]


def time_series_folds(
    snapshots: Sequence[MarketSnapshot],
    *,
    n_splits: int = 1,
    train_ratio: float = 0.70,
) -> tuple[tuple[tuple[MarketSnapshot, ...], tuple[MarketSnapshot, ...]], ...]:
    """Build expanding folds without splitting any ``window_id`` outcome."""

    splits = int(n_splits)
    if splits < 1:
        raise ValueError("n_splits must be positive")
    tape = tuple(snapshots)
    window_groups = _contiguous_window_groups(tape)
    return _window_group_folds(
        window_groups,
        n_splits=splits,
        train_ratio=train_ratio,
    )


def run_grid_search(
    snapshots: Sequence[MarketSnapshot],
    grid: Mapping[str, Sequence[object]],
    *,
    window: MarketWindow,
    windows: Mapping[str, MarketWindow] | None = None,
    base: StrategyBacktestParams | None = None,
    train_ratio: float = 0.70,
    n_splits: int = 1,
    max_workers: int | None = None,
    score: str = "total_return",
    settlement: Mapping[str, float] | None = None,
    spot_prices: Sequence[float] | None = None,
    alpha_books: Sequence[TopOfBook] | None = None,
) -> GridSearchReport:
    """Score combinations on complete-window IS/OOS folds.

    ``max_workers is None`` uses ``os.cpu_count()``. ``max_workers <= 1``
    stays in-process so unit tests do not need a process pool. At least two
    contiguous window groups and metadata for every ``window_id`` are required.
    """

    if score not in _SCORE_NAMES:
        raise ValueError(f"unknown score '{score}'")
    tape = tuple(snapshots)
    if spot_prices is not None and len(spot_prices) != len(tape):
        raise ValueError("spot_prices must match snapshots length")
    if alpha_books is not None and len(alpha_books) != len(tape):
        raise ValueError("alpha_books must match snapshots length")
    params = base if base is not None else StrategyBacktestParams()
    combos = expand_param_grid(grid)
    snapshot_folds = time_series_folds(tape, n_splits=n_splits, train_ratio=train_ratio)
    spots = tuple(float(value) for value in spot_prices) if spot_prices is not None else None
    alpha = tuple(alpha_books) if alpha_books is not None else None
    folds = tuple(
        _aligned_fold(train, test, spot_prices=spots, alpha_books=alpha)
        for train, test in snapshot_folds
    )
    payouts = dict(settlement) if settlement is not None else {}
    window_metadata = dict(windows) if windows is not None else {}
    jobs = tuple(
        _GridJob(
            updates=combo,
            base=params,
            window=window,
            windows=window_metadata,
            folds=folds,
            settlement=payouts,
            score=score,
        )
        for combo in combos
    )
    workers = os.cpu_count() or 1 if max_workers is None else int(max_workers)
    if workers < 1:
        raise ValueError("max_workers must be positive")
    if workers == 1 or len(jobs) == 1:
        trials = tuple(_run_grid_job(job) for job in jobs)
    else:
        ctx = get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            trials = tuple(pool.map(_run_grid_job, jobs))
    best_index = 0
    best_score = trials[0].out_of_sample_score
    for i, trial in enumerate(trials):
        if trial.out_of_sample_score > best_score:
            best_score = trial.out_of_sample_score
            best_index = i
    best = trials[best_index]
    return GridSearchReport(
        score=score,
        best_params=best.params,
        best_trial=best,
        trials=trials,
    )


def _run_grid_job(job: _GridJob) -> GridSearchTrial:
    params = apply_param_updates(job.base, job.updates)
    in_sample: list[StrategyBacktestMetrics] = []
    out_of_sample: list[StrategyBacktestMetrics] = []
    for fold in job.folds:
        if fold.train:
            is_engine = BacktestEngine(
                window=job.window,
                windows=job.windows,
                params=params,
            )
            in_sample.append(
                is_engine.run(
                    fold.train,
                    settlement=job.settlement,
                    spot_prices=fold.train_spots,
                    alpha_books=fold.train_alpha,
                ).metrics
            )
        if fold.test:
            oos_engine = BacktestEngine(
                window=job.window,
                windows=job.windows,
                params=params,
            )
            out_of_sample.append(
                oos_engine.run(
                    fold.test,
                    settlement=job.settlement,
                    spot_prices=fold.test_spots,
                    alpha_books=fold.test_alpha,
                ).metrics
            )
    return GridSearchTrial(
        params=dict(job.updates),
        in_sample_score=_mean_score(in_sample, job.score),
        out_of_sample_score=_mean_score(out_of_sample, job.score),
        in_sample=tuple(in_sample),
        out_of_sample=tuple(out_of_sample),
    )


def _aligned_fold(
    train: tuple[MarketSnapshot, ...],
    test: tuple[MarketSnapshot, ...],
    *,
    spot_prices: tuple[float, ...] | None,
    alpha_books: tuple[TopOfBook, ...] | None,
) -> _GridFold:
    train_end = len(train)
    test_end = train_end + len(test)
    return _GridFold(
        train=train,
        test=test,
        train_spots=None if spot_prices is None else spot_prices[:train_end],
        test_spots=None if spot_prices is None else spot_prices[train_end:test_end],
        train_alpha=None if alpha_books is None else alpha_books[:train_end],
        test_alpha=None if alpha_books is None else alpha_books[train_end:test_end],
    )


def _contiguous_window_groups(
    tape: tuple[MarketSnapshot, ...],
) -> tuple[tuple[MarketSnapshot, ...], ...]:
    groups: list[list[MarketSnapshot]] = []
    seen: set[str] = set()
    current_window: str | None = None
    for snapshot in tape:
        if snapshot.window_id != current_window:
            if snapshot.window_id in seen:
                raise ValueError("each window_id must occupy one contiguous block")
            seen.add(snapshot.window_id)
            current_window = snapshot.window_id
            groups.append([])
        groups[-1].append(snapshot)
    return tuple(tuple(group) for group in groups)


def _window_group_folds(
    groups: tuple[tuple[MarketSnapshot, ...], ...],
    *,
    n_splits: int,
    train_ratio: float,
) -> tuple[tuple[tuple[MarketSnapshot, ...], tuple[MarketSnapshot, ...]], ...]:
    if not 0.0 < float(train_ratio) < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    if len(groups) < n_splits + 1:
        raise ValueError("not enough complete windows for the requested n_splits")
    if n_splits == 1:
        tape = _flatten_window_groups(groups)
        cut_ts = tape[0].timestamp_ms + int(
            (tape[-1].timestamp_ms - tape[0].timestamp_ms) * train_ratio
        )
        train_group_count = 1
        for count in range(1, len(groups)):
            if groups[count - 1][-1].timestamp_ms <= cut_ts:
                train_group_count = count
            else:
                break
        train_group_count = min(train_group_count, len(groups) - 1)
        return (
            (
                _flatten_window_groups(groups[:train_group_count]),
                _flatten_window_groups(groups[train_group_count:]),
            ),
        )

    fold_size = max(1, len(groups) // (n_splits + 1))
    folds: list[tuple[tuple[MarketSnapshot, ...], tuple[MarketSnapshot, ...]]] = []
    for i in range(n_splits):
        train_end = fold_size * (i + 1)
        test_end = min(len(groups), train_end + fold_size)
        if i == n_splits - 1:
            test_end = len(groups)
        train = _flatten_window_groups(groups[:train_end])
        test = _flatten_window_groups(groups[train_end:test_end])
        if train and test:
            folds.append((train, test))
    if not folds:
        raise ValueError("complete-window split produced no usable folds")
    return tuple(folds)


def _flatten_window_groups(
    groups: Sequence[tuple[MarketSnapshot, ...]],
) -> tuple[MarketSnapshot, ...]:
    return tuple(snapshot for group in groups for snapshot in group)


def _mean_score(rows: Sequence[StrategyBacktestMetrics], score: str) -> float:
    if not rows:
        return float("-inf")
    total = 0.0
    for row in rows:
        total += float(getattr(row, score))
    return total / len(rows)
