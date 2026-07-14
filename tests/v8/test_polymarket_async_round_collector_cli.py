"""Async Polymarket round collector CLI tests."""

from __future__ import annotations

import json
from pathlib import Path

from examples.v8.run_polymarket_async_round_collector import (
    _round_start_alignment_sleep_seconds,
    _scheduled_round_start_epoch_seconds,
    main,
)


def test_round_start_alignment_does_not_sleep_inside_start_window() -> None:
    assert (
        _round_start_alignment_sleep_seconds(
            market_family="btc_updown_5m",
            max_round_start_lag_seconds=30.0,
            now_epoch_seconds=600.0 + 12.0,
        )
        == 0.0
    )


def test_round_start_alignment_waits_when_started_late_in_round() -> None:
    assert (
        _round_start_alignment_sleep_seconds(
            market_family="btc_updown_5m",
            max_round_start_lag_seconds=30.0,
            now_epoch_seconds=600.0 + 247.0,
        )
        == 54.0
    )


def test_round_capture_schedule_advances_one_boundary_while_prior_capture_runs() -> None:
    first_start = _scheduled_round_start_epoch_seconds(
        market_family="btc_updown_5m",
        max_round_start_lag_seconds=30.0,
        now_epoch_seconds=600.0 + 12.0,
        previous_round_start_epoch_seconds=None,
    )
    second_start = _scheduled_round_start_epoch_seconds(
        market_family="btc_updown_5m",
        max_round_start_lag_seconds=30.0,
        now_epoch_seconds=600.0 + 13.0,
        previous_round_start_epoch_seconds=first_start,
    )

    assert first_start == 600.0
    assert second_start == 900.0


def test_round_capture_schedule_targets_next_boundary_before_prior_capture_finishes() -> None:
    second_start = _scheduled_round_start_epoch_seconds(
        market_family="btc_updown_5m",
        max_round_start_lag_seconds=30.0,
        now_epoch_seconds=899.5,
        previous_round_start_epoch_seconds=600.0,
    )

    assert second_start == 900.0


def test_finalize_only_cli_accepts_shared_collector_args(tmp_path: Path) -> None:
    assert (
        main(
            [
                "--batch-id",
                "finalize-smoke",
                "--output-dir",
                str(tmp_path),
                "--finalize-only",
                "--max-round-start-lag-seconds",
                "30",
            ]
        )
        == 0
    )

    summary_path = tmp_path / "finalize-smoke" / "finalizer_summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["finalize_only"] is True
    assert summary["finalization_attempt_count"] == 0
    assert summary["error_count"] == 0
