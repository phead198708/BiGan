"""Shared contracts for SELL_BEFORE_CLOSE source-candidate diagnostics."""

from __future__ import annotations

SELL_BEFORE_CLOSE_ONLY_SOURCE_CANDIDATE_NAME = (
    "I_sell_before_close_only_source_candidate"
)
SELL_BEFORE_CLOSE_EXIT_RELIABILITY_GUARD_CANDIDATE_NAME = (
    "J_sell_before_close_exit_reliability_guard_candidate"
)
SELL_BEFORE_CLOSE_ONLY_SOURCE_CANDIDATE_ACTIONS = (
    "BUY_UP_SELL_BEFORE_CLOSE",
    "BUY_DOWN_SELL_BEFORE_CLOSE",
    "NO_TRADE",
)
SELL_BEFORE_CLOSE_DISABLED_SOURCE_CANDIDATE_ACTIONS = (
    "BUY_UP_HOLD_TO_SETTLEMENT",
    "BUY_DOWN_HOLD_TO_SETTLEMENT",
)
SELL_BEFORE_CLOSE_EXIT_RELIABILITY_GUARD_EXIT_POLICY = (
    "first_executable_exit_after_entry"
)
SELL_BEFORE_CLOSE_EXIT_RELIABILITY_GUARD_THRESHOLDS = {
    "min_seconds_to_close": 60.0,
    "min_executable_bid_notional": 0.20,
    "min_queue_fill_probability_proxy": 0.50,
    "max_spread": 1000.0,
    "max_book_staleness_ms": 10_000.0,
    "min_recent_book_update_count_1m": 1.0,
    "min_best_action_margin": 0.0,
    "min_calibrated_action_score": 0.015,
}
