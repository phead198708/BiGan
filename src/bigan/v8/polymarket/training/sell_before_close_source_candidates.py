"""Shared contracts for SELL_BEFORE_CLOSE source-candidate diagnostics."""

from __future__ import annotations

SELL_BEFORE_CLOSE_ONLY_SOURCE_CANDIDATE_NAME = (
    "I_sell_before_close_only_source_candidate"
)
SELL_BEFORE_CLOSE_EXIT_RELIABILITY_GUARD_CANDIDATE_NAME = (
    "J_sell_before_close_exit_reliability_guard_candidate"
)
SELL_BEFORE_CLOSE_P_UP_ALIGNED_GUARD_CANDIDATE_NAME = (
    "K_sell_before_close_exit_reliability_p_up_aligned_candidate"
)
SELL_BEFORE_CLOSE_SUPPORT_AWARE_P_UP_ALIGNED_CANDIDATE_NAME = (
    "L_sell_before_close_support_aware_p_up_aligned_candidate"
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
SELL_BEFORE_CLOSE_P_UP_ALIGNED_GUARD_THRESHOLDS = {
    "p_up_alignment_min": 0.55,
    "min_seconds_to_close": 90.0,
    "min_executable_bid_notional": 0.20,
    "min_queue_fill_probability_proxy": 0.65,
    "max_spread": 900.0,
    "max_book_staleness_ms": 10_000.0,
    "min_recent_book_update_count_1m": 1.0,
    "min_best_action_margin": 0.01,
    "min_calibrated_action_score": 0.03,
    "max_entries_per_market": 1.0,
    "min_reentry_cooldown_seconds": 60.0,
}
SELL_BEFORE_CLOSE_GUARD_THRESHOLD_SWEEP_GRID = {
    "p_up_alignment_min": (0.50, 0.55, 0.60, 0.65),
    "min_calibrated_action_score": (0.015, 0.03, 0.05),
    "min_best_action_margin": (0.0, 0.01, 0.02),
    "min_queue_fill_probability_proxy": (0.50, 0.65, 0.80),
}
SELL_BEFORE_CLOSE_SUPPORT_AWARE_THRESHOLD_SELECTION_GRID = {
    "p_up_alignment_min": (0.50, 0.52, 0.55, 0.58, 0.60),
    "min_calibrated_action_score": (0.015, 0.02, 0.03, 0.04, 0.05),
    "min_best_action_margin": (0.0, 0.005, 0.01, 0.02),
    "min_queue_fill_probability_proxy": (0.50, 0.60, 0.65, 0.75),
    "min_seconds_to_close": (60.0, 75.0, 90.0, 120.0),
    "max_entries_per_market": (1.0, 2.0),
    "min_reentry_cooldown_seconds": (60.0, 90.0, 120.0),
}
