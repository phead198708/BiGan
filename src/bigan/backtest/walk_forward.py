"""Walk-forward backtest orchestration (issue #14)."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class WalkForwardWindow:
    """One strictly time-ordered train/validation/test slice."""

    window_id: str
    train_start_ts: int
    train_end_ts: int
    val_start_ts: int
    val_end_ts: int
    test_start_ts: int
    test_end_ts: int

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WalkForwardWindowResult:
    """Metrics produced by one walk-forward window."""

    window: WalkForwardWindow
    metrics: dict[str, float | int | None]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "window": self.window.to_dict(),
            "metrics": self.metrics,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class MetricStability:
    """Cross-window stability summary for one metric."""

    metric: str
    count: int
    mean: float | None
    minimum: float | None
    maximum: float | None
    stddev: float | None

    def to_dict(self) -> dict[str, float | int | str | None]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WalkForwardReport:
    """All window-level results plus cross-window summaries."""

    windows: tuple[WalkForwardWindowResult, ...]
    metric_stability: tuple[MetricStability, ...]

    @property
    def window_count(self) -> int:
        return len(self.windows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_count": self.window_count,
            "windows": [window.to_dict() for window in self.windows],
            "metric_stability": [metric.to_dict() for metric in self.metric_stability],
        }


def generate_walk_forward_windows(
    *,
    start_ts: int,
    end_ts: int,
    train_window_ms: int,
    val_window_ms: int,
    test_window_ms: int,
    step_ms: int | None = None,
) -> tuple[WalkForwardWindow, ...]:
    """Generate deterministic rolling train/val/test windows.

    ``train`` immediately precedes ``val`` and ``val`` immediately precedes
    ``test``. The start of each next window advances by ``step_ms``; by
    default this equals ``test_window_ms``.
    """

    if end_ts <= start_ts:
        raise ValueError("end_ts must be greater than start_ts")
    for name, value in (
        ("train_window_ms", train_window_ms),
        ("val_window_ms", val_window_ms),
        ("test_window_ms", test_window_ms),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    step = test_window_ms if step_ms is None else step_ms
    if step <= 0:
        raise ValueError("step_ms must be positive")

    windows: list[WalkForwardWindow] = []
    train_start = start_ts
    idx = 0
    while True:
        train_end = train_start + train_window_ms
        val_start = train_end
        val_end = val_start + val_window_ms
        test_start = val_end
        test_end = test_start + test_window_ms
        if test_end > end_ts:
            break
        windows.append(
            WalkForwardWindow(
                window_id=f"wf-{idx:04d}",
                train_start_ts=train_start,
                train_end_ts=train_end,
                val_start_ts=val_start,
                val_end_ts=val_end,
                test_start_ts=test_start,
                test_end_ts=test_end,
            )
        )
        idx += 1
        train_start += step
    return tuple(windows)


def run_walk_forward(
    windows: Sequence[WalkForwardWindow],
    run_window: Callable[[WalkForwardWindow], Mapping[str, float | int | None]],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> WalkForwardReport:
    """Run a caller-supplied train/evaluate callback for each time window."""

    results: list[WalkForwardWindowResult] = []
    for window in windows:
        metrics = dict(run_window(window))
        results.append(
            WalkForwardWindowResult(
                window=window,
                metrics=metrics,
                metadata=dict(metadata or {}),
            )
        )
    return WalkForwardReport(
        windows=tuple(results),
        metric_stability=summarize_metric_stability(results),
    )


def summarize_metric_stability(
    results: Sequence[WalkForwardWindowResult],
) -> tuple[MetricStability, ...]:
    """Summarize numeric metrics across windows."""

    metric_names = sorted({name for result in results for name in result.metrics})
    summaries: list[MetricStability] = []
    for name in metric_names:
        values = [
            float(value)
            for result in results
            if (value := result.metrics.get(name)) is not None
        ]
        if not values:
            summaries.append(
                MetricStability(
                    metric=name,
                    count=0,
                    mean=None,
                    minimum=None,
                    maximum=None,
                    stddev=None,
                )
            )
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        summaries.append(
            MetricStability(
                metric=name,
                count=len(values),
                mean=mean,
                minimum=min(values),
                maximum=max(values),
                stddev=math.sqrt(variance),
            )
        )
    return tuple(summaries)


def save_walk_forward_report(report: WalkForwardReport, path: Path | str) -> None:
    """Persist the full walk-forward report as JSON."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
