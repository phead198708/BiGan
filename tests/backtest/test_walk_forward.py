"""Walk-forward workflow tests for issue #14."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bigan.backtest import (
    WalkForwardWindowResult,
    generate_walk_forward_windows,
    run_walk_forward,
    save_walk_forward_report,
    summarize_metric_stability,
)


def test_generate_walk_forward_windows_uses_time_ordered_slices() -> None:
    windows = generate_walk_forward_windows(
        start_ts=0,
        end_ts=21_000,
        train_window_ms=10_000,
        val_window_ms=2_000,
        test_window_ms=3_000,
    )

    assert [window.window_id for window in windows] == ["wf-0000", "wf-0001", "wf-0002"]
    assert windows[0].train_start_ts == 0
    assert windows[0].train_end_ts == 10_000
    assert windows[0].val_start_ts == 10_000
    assert windows[0].val_end_ts == 12_000
    assert windows[0].test_start_ts == 12_000
    assert windows[0].test_end_ts == 15_000
    assert windows[1].train_start_ts == 3_000
    for window in windows:
        assert window.train_start_ts < window.train_end_ts
        assert window.train_end_ts == window.val_start_ts
        assert window.val_end_ts == window.test_start_ts
        assert window.test_start_ts < window.test_end_ts


def test_run_walk_forward_tracks_each_window_and_summarizes_metrics() -> None:
    windows = generate_walk_forward_windows(
        start_ts=0,
        end_ts=21_000,
        train_window_ms=10_000,
        val_window_ms=2_000,
        test_window_ms=3_000,
    )

    def run_window(window):
        index = int(window.window_id.rsplit("-", 1)[1])
        return {"net_return": 0.01 * (index + 1), "trade_count": index + 2}

    report = run_walk_forward(windows, run_window, metadata={"model_version": "xgb-v1"})

    assert report.window_count == 3
    assert report.windows[0].window.window_id == "wf-0000"
    assert report.windows[0].metadata == {"model_version": "xgb-v1"}
    by_metric = {metric.metric: metric for metric in report.metric_stability}
    assert by_metric["net_return"].count == 3
    assert by_metric["net_return"].mean == pytest.approx(0.02)
    assert by_metric["net_return"].minimum == pytest.approx(0.01)
    assert by_metric["net_return"].maximum == pytest.approx(0.03)
    assert by_metric["trade_count"].mean == pytest.approx(3.0)


def test_summarize_metric_stability_handles_missing_metrics() -> None:
    windows = generate_walk_forward_windows(
        start_ts=0,
        end_ts=15_000,
        train_window_ms=10_000,
        val_window_ms=2_000,
        test_window_ms=3_000,
    )
    results = [
        WalkForwardWindowResult(
            window=windows[0],
            metrics={"net_return": None, "trade_count": 2},
            metadata={},
        )
    ]

    summary = {metric.metric: metric for metric in summarize_metric_stability(results)}

    assert summary["net_return"].count == 0
    assert summary["net_return"].mean is None
    assert summary["trade_count"].mean == pytest.approx(2.0)


def test_walk_forward_report_can_be_saved(tmp_path: Path) -> None:
    windows = generate_walk_forward_windows(
        start_ts=0,
        end_ts=15_000,
        train_window_ms=10_000,
        val_window_ms=2_000,
        test_window_ms=3_000,
    )
    report = run_walk_forward(windows, lambda _: {"net_return": 0.1})
    path = tmp_path / "reports" / "walk-forward.json"

    save_walk_forward_report(report, path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["window_count"] == 1
    assert data["windows"][0]["window"]["window_id"] == "wf-0000"
    assert data["metric_stability"][0]["metric"] == "net_return"
