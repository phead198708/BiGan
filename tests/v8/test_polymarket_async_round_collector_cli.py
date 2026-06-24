"""Async Polymarket round collector CLI tests."""

from __future__ import annotations

from examples.v8.run_polymarket_async_round_collector import (
    _round_start_alignment_sleep_seconds,
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
