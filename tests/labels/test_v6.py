from __future__ import annotations

import pytest

from bigan.labels.v6 import (
    VolatilityLabelConfig,
    compute_volatility_path_label,
    settlement_3way_label,
    two_sided_volatility_fields,
)


def test_settlement_3way_label_uses_direction_with_neutral_margin() -> None:
    assert settlement_3way_label(100.0, 101.0, neutral_margin_abs=0.05) == "UP"
    assert settlement_3way_label(100.0, 99.0, neutral_margin_abs=0.05) == "DOWN"
    assert settlement_3way_label(100.0, 100.03, neutral_margin_abs=0.05) == "NEUTRAL"


def test_compute_volatility_path_label_uses_bid_ask_slippage_and_strict_future_path() -> None:
    base = 1_780_000_000_000
    result = compute_volatility_path_label(
        [
            {"ts": base - 1_000, "bid": 0.95, "ask": 0.96},
            {"ts": base, "bid": 0.39, "ask": 0.40},
            {"ts": base, "bid": 0.90, "ask": 0.91},
            {"ts": base + 120_000, "bid": 0.60, "ask": 0.61},
        ],
        decision_ts=base,
        round_end_ts=base + 900_000,
        config=VolatilityLabelConfig(
            min_exit_gain=0.15,
            buy_slippage=0.02,
            sell_slippage=0.02,
            min_exit_seconds_before_expiry=300.0,
            max_entry_wait_ms=60_000,
            min_entry_price=0.35,
        ),
    )

    assert result.path_validity_flag == "valid"
    assert result.entry_worst_price == pytest.approx(0.42)
    assert result.best_exit_bid == pytest.approx(0.60)
    assert result.best_exit_price == pytest.approx(0.58)
    assert result.max_exit_gain == pytest.approx(0.16)
    assert result.time_to_best_exit_seconds == pytest.approx(120.0)
    assert result.label is True


def test_two_sided_volatility_fields_reports_independent_side_coverage() -> None:
    base = 1_780_000_000_000
    fields = two_sided_volatility_fields(
        quotes_by_side={
            "UP": [
                {"ts": base, "bid": 0.39, "ask": 0.40},
                {"ts": base + 120_000, "bid": 0.60, "ask": 0.61},
            ],
            "DOWN": [],
        },
        decision_ts=base,
        round_end_ts=base + 900_000,
        config=VolatilityLabelConfig(min_exit_gain=0.15),
    )

    assert fields["label_volatility_up"] is True
    assert fields["max_exit_gain_up"] == pytest.approx(0.16)
    assert fields["label_volatility_down"] is None
    assert fields["volatility_path_validity_down"] == "missing_entry_quote"


def test_compute_volatility_path_label_deducts_entry_and_exit_fees() -> None:
    base = 1_780_000_000_000
    result = compute_volatility_path_label(
        [
            {"ts": base, "bid": 0.39, "ask": 0.40},
            {"ts": base + 120_000, "bid": 0.60, "ask": 0.61},
        ],
        decision_ts=base,
        round_end_ts=base + 900_000,
        config=VolatilityLabelConfig(
            min_exit_gain=0.15,
            buy_slippage=0.02,
            sell_slippage=0.02,
            fee_bps=100.0,
        ),
    )

    assert result.entry_worst_price == pytest.approx(0.424)
    assert result.best_exit_price == pytest.approx(0.574)
    assert result.max_exit_gain == pytest.approx(0.15)
