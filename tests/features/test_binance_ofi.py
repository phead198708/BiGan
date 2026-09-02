"""DEV-01 Binance L2 top-of-book OFI engine."""

from __future__ import annotations

import math
import random
import time

import pytest

from bigan.features.binance_ofi import (
    BinanceOFICalculator,
    cont_ask_imbalance,
    cont_bid_imbalance,
)


def _hand_raw(prev: tuple[float, float, float, float], curr: tuple[float, float, float, float]) -> tuple[float, float, float]:
    i_b = cont_bid_imbalance(prev[0], prev[1], curr[0], curr[1])
    i_a = cont_ask_imbalance(prev[2], prev[3], curr[2], curr[3])
    return i_b, i_a, i_b - i_a


def _seed(calc: BinanceOFICalculator, book: tuple[float, float, float, float], ts_ms: int = 1_000) -> None:
    assert (
        calc.on_depth_update(
            bid_price=book[0],
            bid_qty=book[1],
            ask_price=book[2],
            ask_qty=book[3],
            ts_ms=ts_ms,
        )
        is None
    )


def _step(
    calc: BinanceOFICalculator,
    book: tuple[float, float, float, float],
    ts_ms: int,
):
    snapshot = calc.on_depth_update(
        bid_price=book[0],
        bid_qty=book[1],
        ask_price=book[2],
        ask_qty=book[3],
        ts_ms=ts_ms,
    )
    assert snapshot is not None
    return snapshot


SCENARIOS = {
    "price_up": {
        "prev": (100.00, 2.0, 100.01, 3.0),
        "curr": (100.01, 1.5, 100.02, 2.5),
        # Bid lifted → I_b = +q_b(t) = 1.5
        # Ask lifted → I_a = -q_a(t-1) = -3.0
        # raw_ofi = 1.5 - (-3.0) = 4.5
        "i_b": 1.5,
        "i_a": -3.0,
        "raw_ofi": 4.5,
    },
    "flat_bid_depth_down": {
        "prev": (100.00, 5.0, 100.01, 4.0),
        "curr": (100.00, 3.0, 100.01, 4.0),
        # Same prices, bid size 5 → 3 → I_b = -2
        # Ask unchanged → I_a = 0
        "i_b": -2.0,
        "i_a": 0.0,
        "raw_ofi": -2.0,
    },
    "ask_cancel": {
        "prev": (100.00, 2.0, 100.01, 6.0),
        "curr": (100.00, 2.0, 100.01, 1.0),
        # Ask size 6 → 1 at same price → I_a = -5
        "i_b": 0.0,
        "i_a": -5.0,
        "raw_ofi": 5.0,
    },
    "dump": {
        "prev": (100.00, 4.0, 100.01, 2.0),
        "curr": (99.99, 3.0, 100.00, 5.0),
        # Bid dropped → I_b = -q_b(t-1) = -4
        # Ask dropped → I_a = +q_a(t) = 5
        # raw_ofi = -4 - 5 = -9
        "i_b": -4.0,
        "i_a": 5.0,
        "raw_ofi": -9.0,
    },
    "bid_add": {
        "prev": (100.00, 2.0, 100.01, 2.0),
        "curr": (100.00, 8.0, 100.01, 2.0),
        # Same prices, bid size 2 → 8 → I_b = +6
        "i_b": 6.0,
        "i_a": 0.0,
        "raw_ofi": 6.0,
    },
}


def test_five_orderbook_sequences_match_hand_calculated_raw_ofi() -> None:
    assert len(SCENARIOS) == 5
    for name, spec in SCENARIOS.items():
        calc = BinanceOFICalculator()
        _seed(calc, spec["prev"], ts_ms=1_000)
        snapshot = _step(calc, spec["curr"], ts_ms=1_001)
        expected_i_b, expected_i_a, expected_raw = _hand_raw(spec["prev"], spec["curr"])
        assert expected_i_b == spec["i_b"], name
        assert expected_i_a == spec["i_a"], name
        assert expected_raw == spec["raw_ofi"], name
        assert snapshot.i_b == pytest.approx(spec["i_b"]), name
        assert snapshot.i_a == pytest.approx(spec["i_a"]), name
        assert snapshot.raw_ofi == pytest.approx(spec["raw_ofi"]), name
        assert calc.last_raw_ofi == pytest.approx(spec["raw_ofi"]), name


def test_consecutive_stream_preserves_each_hand_calculated_step() -> None:
    books = [
        (100.00, 2.0, 100.01, 3.0),
        (100.01, 1.5, 100.02, 2.5),  # price_up
        (100.01, 1.0, 100.02, 2.5),  # flat bid down
        (100.01, 1.0, 100.02, 0.5),  # ask cancel
        (100.00, 4.0, 100.01, 6.0),  # dump
        (100.00, 9.0, 100.01, 6.0),  # bid add
    ]
    calc = BinanceOFICalculator()
    _seed(calc, books[0], ts_ms=10_000)
    for index, book in enumerate(books[1:], start=1):
        snapshot = _step(calc, book, ts_ms=10_000 + index)
        i_b, i_a, raw_ofi = _hand_raw(books[index - 1], book)
        assert snapshot.i_b == pytest.approx(i_b)
        assert snapshot.i_a == pytest.approx(i_a)
        assert snapshot.raw_ofi == pytest.approx(raw_ofi)


def test_ema_converges_on_repeated_same_signed_events_without_nan() -> None:
    calc = BinanceOFICalculator(ema_alpha=0.25)
    bid_qty = 1.0
    _seed(calc, (100.00, bid_qty, 100.01, 1.0), ts_ms=0)
    ema_values: list[float] = []
    for step in range(1, 41):
        bid_qty += 1.0
        snapshot = _step(
            calc,
            (100.00, bid_qty, 100.01, 1.0),
            ts_ms=step,
        )
        assert snapshot.raw_ofi == pytest.approx(1.0)
        assert math_isfinite(snapshot.ema_ofi)
        ema_values.append(snapshot.ema_ofi)
    # First event is +1, so EMA is already 1 and stays there.
    assert ema_values[0] == pytest.approx(1.0)
    assert ema_values[-1] == pytest.approx(1.0)
    assert all(value == pytest.approx(1.0) for value in ema_values)

    calc = BinanceOFICalculator(ema_alpha=0.25)
    _seed(calc, (100.00, 1.0, 100.01, 1.0), ts_ms=0)
    # Large first event, then a constant +1 flow: EMA must decay toward 1.
    first = _step(calc, (100.00, 11.0, 100.01, 1.0), ts_ms=1)  # raw = +10
    assert first.raw_ofi == pytest.approx(10.0)
    assert first.ema_ofi == pytest.approx(10.0)
    ema_path = [first.ema_ofi]
    bid_qty = 11.0
    for step in range(2, 32):
        bid_qty += 1.0
        snapshot = _step(calc, (100.00, bid_qty, 100.01, 1.0), ts_ms=step)
        assert snapshot.raw_ofi == pytest.approx(1.0)
        ema_path.append(snapshot.ema_ofi)
        assert math_isfinite(snapshot.ema_ofi)
    assert ema_path[1] < ema_path[0]
    assert ema_path[-1] < ema_path[5]
    assert ema_path[-1] == pytest.approx(1.0, abs=0.05)
    assert all(
        later <= earlier + 1e-12 for earlier, later in zip(ema_path, ema_path[1:], strict=False)
    )


def test_depth_update_average_latency_under_half_millisecond() -> None:
    calc = BinanceOFICalculator(window_ms=60_000, ema_alpha=0.2)
    _seed(calc, (100.00, 1.0, 100.01, 1.0), ts_ms=0)
    bid_qty = 1.0
    for warmup in range(1, 2_001):
        bid_qty += 0.01
        _step(calc, (100.00, bid_qty, 100.01, 1.0), ts_ms=warmup)
    iterations = 20_000
    started = time.perf_counter()
    for step in range(2_001, 2_001 + iterations):
        bid_qty += 0.01
        calc.on_depth_update(
            bid_price=100.00,
            bid_qty=bid_qty,
            ask_price=100.01,
            ask_qty=1.0,
            ts_ms=step,
        )
    elapsed = time.perf_counter() - started
    average_ms = (elapsed / iterations) * 1_000.0
    assert average_ms < 0.5, f"average depth-update latency {average_ms:.4f}ms"


def test_normalized_ofi_is_zero_until_twenty_samples_then_clipped() -> None:
    calc = BinanceOFICalculator(
        ema_alpha=1.0,
        window_ms=60_000,
        zscore_min_samples=20,
        zscore_clip=3.0,
    )
    _seed(calc, (100.00, 1.0, 100.01, 1.0), ts_ms=0)
    bid_qty = 1.0
    for step in range(1, 20):
        bid_qty += 1.0
        snapshot = _step(calc, (100.00, bid_qty, 100.01, 1.0), ts_ms=step)
        assert snapshot.z_ofi == 0.0
        assert calc.get_normalized_ofi() == 0.0
    snapshot = _step(calc, (100.00, bid_qty + 1.0, 100.01, 1.0), ts_ms=20)
    assert calc.get_normalized_ofi() == snapshot.z_ofi
    # Constant +1 flow → zero variance → safe 0.0, not DivZero.
    assert snapshot.z_ofi == 0.0

    spike = BinanceOFICalculator(ema_alpha=1.0, zscore_min_samples=20, zscore_clip=3.0)
    _seed(spike, (100.00, 10.0, 100.01, 10.0), ts_ms=0)
    bid_qty = 10.0
    for step in range(1, 20):
        bid_qty += 0.1
        _step(spike, (100.00, bid_qty, 100.01, 10.0), ts_ms=step)
        assert spike.get_normalized_ofi() == 0.0
    # One spoof-sized add after a quiet tape.
    huge = _step(spike, (100.00, bid_qty + 10_000.0, 100.01, 10.0), ts_ms=20)
    assert huge.z_ofi == 3.0
    assert spike.get_normalized_ofi() == 3.0
    assert -3.0 <= spike.get_normalized_ofi() <= 3.0

    dump = BinanceOFICalculator(ema_alpha=1.0, zscore_min_samples=20, zscore_clip=3.0)
    _seed(dump, (100.00, 10_000.0, 100.01, 10.0), ts_ms=0)
    bid_qty = 10_000.0
    for step in range(1, 20):
        bid_qty -= 0.1
        _step(dump, (100.00, bid_qty, 100.01, 10.0), ts_ms=step)
    crash = _step(dump, (99.00, 1.0, 99.01, 50_000.0), ts_ms=20)
    assert crash.z_ofi == -3.0
    assert dump.get_normalized_ofi() == -3.0


def test_book_ticker_and_partial_depth_ingest() -> None:
    calc = BinanceOFICalculator(symbol="BTCUSDT")
    first = calc.on_book_ticker(
        {
            "s": "BTCUSDT",
            "E": 1_700_000_000_000,
            "b": "100.00",
            "B": "2.0",
            "a": "100.01",
            "A": "3.0",
        }
    )
    assert first is None
    second = calc.on_partial_depth(
        {
            "lastUpdateId": 2,
            "bids": [["100.01", "1.5"]],
            "asks": [["100.02", "2.5"]],
        },
        ts_ms=1_700_000_000_001,
    )
    assert second is not None
    assert second.raw_ofi == pytest.approx(4.5)


def test_update_and_get_z_matches_snapshot_and_reset_clears_state() -> None:
    snapshot_calc = BinanceOFICalculator(ema_alpha=1.0)
    hot_calc = BinanceOFICalculator(ema_alpha=1.0)
    _seed(snapshot_calc, (100.00, 2.0, 100.01, 3.0), ts_ms=1_000)
    assert (
        hot_calc.update_and_get_z(
            bid_price=100.00,
            bid_qty=2.0,
            ask_price=100.01,
            ask_qty=3.0,
            ts_ms=1_000,
        )
        == 0.0
    )
    snapshot = _step(snapshot_calc, (100.01, 1.5, 100.02, 2.5), ts_ms=1_001)
    z_ofi = hot_calc.update_and_get_z(
        bid_price=100.01,
        bid_qty=1.5,
        ask_price=100.02,
        ask_qty=2.5,
        ts_ms=1_001,
    )
    assert z_ofi == snapshot.z_ofi
    assert hot_calc.last_raw_ofi == pytest.approx(4.5)
    hot_calc.reset()
    assert hot_calc.last_raw_ofi == 0.0
    assert hot_calc.last_ema_ofi == 0.0
    assert hot_calc.get_normalized_ofi() == 0.0
    assert len(hot_calc._samples) == 0
    assert hot_calc._sum == 0.0
    assert hot_calc._sum_sq == 0.0
    assert hot_calc._recalc_counter == 0
    assert (
        hot_calc.update_and_get_z(
            bid_price=100.00,
            bid_qty=2.0,
            ask_price=100.01,
            ask_qty=3.0,
            ts_ms=2_000,
        )
        == 0.0
    )


def test_incremental_moments_match_full_recompute_after_100k_updates() -> None:
    calc = BinanceOFICalculator(
        ema_alpha=0.2,
        window_ms=10_000_000,
        max_events_cap=200_000,
    )
    rng = random.Random(42)
    bid_price = 100.0
    bid_qty = 1.5
    ask_qty = 1.5
    calc.update_and_get_z(
        bid_price=bid_price,
        bid_qty=bid_qty,
        ask_price=bid_price + 0.01,
        ask_qty=ask_qty,
        ts_ms=0,
    )
    for step in range(1, 100_001):
        bid_price = max(1.0, bid_price + rng.uniform(-0.02, 0.02))
        bid_qty = max(0.01, bid_qty + rng.uniform(-0.4, 0.4))
        ask_qty = max(0.01, ask_qty + rng.uniform(-0.4, 0.4))
        calc.update_and_get_z(
            bid_price=bid_price,
            bid_qty=bid_qty,
            ask_price=bid_price + rng.uniform(0.01, 0.05),
            ask_qty=ask_qty,
            ts_ms=step,
        )
    values = [sample[1] for sample in calc._samples]
    expected_sum = sum(values)
    expected_sum_sq = sum(value * value for value in values)
    assert len(values) == 100_000
    assert _relative_error(calc._sum, expected_sum) < 1e-12
    assert _relative_error(calc._sum_sq, expected_sum_sq) < 1e-12


def test_time_window_keeps_120k_ticks_inside_60s() -> None:
    window_ms = 60_000
    tick_count = 120_000
    calc = BinanceOFICalculator(
        ema_alpha=1.0,
        window_ms=window_ms,
        max_events_cap=200_000,
    )
    calc.update_and_get_z(
        bid_price=100.00,
        bid_qty=1.0,
        ask_price=100.01,
        ask_qty=1.0,
        ts_ms=0,
    )
    bid_qty = 1.0
    for index in range(tick_count):
        ts_ms = (index * window_ms) // (tick_count - 1)
        bid_qty += 0.0001
        calc.update_and_get_z(
            bid_price=100.00,
            bid_qty=bid_qty,
            ask_price=100.01,
            ask_qty=1.0,
            ts_ms=ts_ms,
        )
    samples = calc._samples
    assert len(samples) == tick_count
    oldest_ts = samples[0][0]
    newest_ts = samples[-1][0]
    assert newest_ts - oldest_ts == window_ms
    assert newest_ts - oldest_ts == calc.window_ms
    assert oldest_ts >= newest_ts - calc.window_ms


def test_max_events_cap_is_memory_pad_not_time_window() -> None:
    calc = BinanceOFICalculator(window_ms=60_000, max_events_cap=50)
    calc.update_and_get_z(
        bid_price=100.00,
        bid_qty=1.0,
        ask_price=100.01,
        ask_qty=1.0,
        ts_ms=0,
    )
    for step in range(1, 81):
        calc.update_and_get_z(
            bid_price=100.00,
            bid_qty=1.0 + step,
            ask_price=100.01,
            ask_qty=1.0,
            ts_ms=step,
        )
    assert len(calc._samples) == 50
    assert calc._samples[-1][0] - calc._samples[0][0] == 49


def test_variance_numeric_stability_zero_variance() -> None:
    book = (100.00, 2.0, 100.01, 3.0)
    snapshot_calc = BinanceOFICalculator(ema_alpha=1.0, zscore_min_samples=20)
    hot_calc = BinanceOFICalculator(ema_alpha=1.0, zscore_min_samples=20)
    _seed(snapshot_calc, book, ts_ms=0)
    assert (
        hot_calc.update_and_get_z(
            bid_price=book[0],
            bid_qty=book[1],
            ask_price=book[2],
            ask_qty=book[3],
            ts_ms=0,
        )
        == 0.0
    )
    for step in range(1, 101):
        snapshot = _step(snapshot_calc, book, ts_ms=step)
        z_ofi = hot_calc.update_and_get_z(
            bid_price=book[0],
            bid_qty=book[1],
            ask_price=book[2],
            ask_qty=book[3],
            ts_ms=step,
        )
        assert snapshot.raw_ofi == 0.0
        assert snapshot.z_ofi == 0.0
        assert z_ofi == 0.0
        assert snapshot_calc.get_normalized_ofi() == 0.0
        assert hot_calc.get_normalized_ofi() == 0.0


def test_ticker_float_missing_keys() -> None:
    calc = BinanceOFICalculator(symbol="BTCUSDT")
    with pytest.raises(ValueError, match="missing both 'b' and 'bidPrice'"):
        calc.on_book_ticker({"s": "BTCUSDT", "E": 1})
    with pytest.raises(ValueError, match="missing both 'b' and 'bidPrice'"):
        calc.on_book_ticker({})
    with pytest.raises(ValueError, match="missing both 'B' and 'bidQty'"):
        calc.on_book_ticker({"s": "BTCUSDT", "E": 1, "b": "100.00", "a": "100.01", "A": "1.0"})


def test_recalibration_precision() -> None:
    calc = BinanceOFICalculator(
        ema_alpha=0.2,
        window_ms=10_000_000,
        max_events_cap=20_000,
    )
    bid_price = 100.0
    bid_qty = 1.0 / 3.0
    ask_qty = 2.0 / 7.0
    calc.update_and_get_z(
        bid_price=bid_price,
        bid_qty=bid_qty,
        ask_price=bid_price + math.pi * 1e-4,
        ask_qty=ask_qty,
        ts_ms=0,
    )
    for step in range(1, 10_001):
        bid_price = max(1.0, bid_price + math.sin(step / 17.0) * 1e-6)
        bid_qty = max(1e-9, bid_qty + math.cos(step / 13.0) * 1e-7)
        ask_qty = max(1e-9, ask_qty + math.sin(step / 11.0) * 1e-7)
        calc.update_and_get_z(
            bid_price=bid_price,
            bid_qty=bid_qty,
            ask_price=bid_price + math.pi * 1e-4,
            ask_qty=ask_qty,
            ts_ms=step,
        )
    values = [sample[1] for sample in calc._samples]
    expected_sum = sum(values)
    expected_sum_sq = sum(value * value for value in values)
    assert len(values) == 10_000
    assert calc._recalc_counter == 0
    assert _relative_error(calc._sum, expected_sum) < 1e-12
    assert _relative_error(calc._sum_sq, expected_sum_sq) < 1e-12


def test_update_and_get_z_equivalence() -> None:
    sequence = [
        (100.00, 2.0, 100.01, 3.0),
        (100.01, 1.5, 100.02, 2.5),
        (100.01, 1.0, 100.02, 2.5),
        (100.01, 1.0, 100.02, 0.5),
        (100.00, 4.0, 100.01, 6.0),
        (100.00, 9.0, 100.01, 6.0),
        (99.99, 3.0, 100.00, 8.0),
        (99.99, 3.2, 100.00, 7.4),
    ]
    snapshot_calc = BinanceOFICalculator(ema_alpha=0.2, zscore_min_samples=3)
    hot_calc = BinanceOFICalculator(ema_alpha=0.2, zscore_min_samples=3)
    _seed(snapshot_calc, sequence[0], ts_ms=0)
    assert (
        hot_calc.update_and_get_z(
            bid_price=sequence[0][0],
            bid_qty=sequence[0][1],
            ask_price=sequence[0][2],
            ask_qty=sequence[0][3],
            ts_ms=0,
        )
        == 0.0
    )
    for index, book in enumerate(sequence[1:], start=1):
        snapshot = _step(snapshot_calc, book, ts_ms=index)
        z_ofi = hot_calc.update_and_get_z(
            bid_price=book[0],
            bid_qty=book[1],
            ask_price=book[2],
            ask_qty=book[3],
            ts_ms=index,
        )
        assert z_ofi == snapshot.z_ofi
        assert hot_calc.last_raw_ofi == snapshot.raw_ofi
        assert hot_calc.last_ema_ofi == snapshot.ema_ofi


def math_isfinite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def _relative_error(actual: float, expected: float) -> float:
    return abs(actual - expected) / max(abs(expected), 1e-15)
