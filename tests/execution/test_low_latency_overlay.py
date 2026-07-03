from __future__ import annotations

from pathlib import Path

import pytest

from bigan.execution.low_latency_overlay import (
    LowLatencyEntryOverlay,
    LowLatencyOverlayConfig,
)
from bigan.features.low_latency import JsonlRawQueue


def _signal(*, market_implied_prob: float = 0.50) -> dict:
    return {
        "canonical_symbol": "BTC-15M:btc-updown-15m-1780803000:UP",
        "market_implied_prob": market_implied_prob,
        "outcome_side": "UP",
        "token_probability": 0.82,
    }


def _append_tob(
    queue_path: Path,
    *,
    ts: int,
    bid: float,
    ask: float,
    canonical_symbol: str = "BTC-15M:btc-updown-15m-1780803000:UP",
) -> None:
    JsonlRawQueue(queue_path).append(
        "raw_top_of_book",
        {
            "canonical_symbol": canonical_symbol,
            "source": "polymarket",
            "source_symbol": "token-up",
            "source_market": "0xmkt",
            "ts": ts,
            "message_ts": ts,
            "capture_timestamp_ms": ts,
            "bid_price": bid,
            "ask_price": ask,
            "spread": ask - bid,
        },
        published_at_ms=ts + 1,
    )


def test_low_latency_overlay_passes_fresh_stable_quote(tmp_path: Path) -> None:
    queue_path = tmp_path / "raw.jsonl"
    _append_tob(queue_path, ts=10_000, bid=0.48, ask=0.50)
    _append_tob(queue_path, ts=18_000, bid=0.51, ask=0.53)
    overlay = LowLatencyEntryOverlay(
        queue_path,
        config=LowLatencyOverlayConfig(enabled=True),
    )
    report = overlay.refresh()

    decision = overlay.evaluate_entry(_signal(market_implied_prob=0.50), now_ms=19_000)

    assert report.top_of_book_rows_applied == 2
    assert decision.passed is True
    assert decision.reason == "overlay_pass"
    assert decision.mid_velocity == pytest.approx(0.03)
    assert decision.price_drift_from_signal == pytest.approx(0.03)


def test_low_latency_overlay_skips_adverse_side_velocity(tmp_path: Path) -> None:
    queue_path = tmp_path / "raw.jsonl"
    _append_tob(queue_path, ts=10_000, bid=0.52, ask=0.54)
    _append_tob(queue_path, ts=18_000, bid=0.61, ask=0.63)
    overlay = LowLatencyEntryOverlay(
        queue_path,
        config=LowLatencyOverlayConfig(
            enabled=True,
            adverse_velocity_threshold=0.04,
            max_price_drift_from_signal=None,
        ),
    )
    overlay.refresh()

    decision = overlay.evaluate_entry(_signal(market_implied_prob=0.62), now_ms=19_000)

    assert decision.passed is False
    assert decision.reason == "overlay_adverse_side_velocity"
    assert decision.mid_velocity == pytest.approx(0.09)


def test_low_latency_overlay_allows_favorable_side_velocity(tmp_path: Path) -> None:
    queue_path = tmp_path / "raw.jsonl"
    _append_tob(queue_path, ts=10_000, bid=0.61, ask=0.63)
    _append_tob(queue_path, ts=18_000, bid=0.52, ask=0.54)
    overlay = LowLatencyEntryOverlay(
        queue_path,
        config=LowLatencyOverlayConfig(
            enabled=True,
            adverse_velocity_threshold=0.04,
        ),
    )
    overlay.refresh()

    decision = overlay.evaluate_entry(_signal(market_implied_prob=0.52), now_ms=19_000)

    assert decision.passed is True
    assert decision.reason == "overlay_pass"
    assert decision.mid_velocity == pytest.approx(-0.09)


def test_low_latency_overlay_skips_price_drift_from_signal(tmp_path: Path) -> None:
    queue_path = tmp_path / "raw.jsonl"
    _append_tob(queue_path, ts=18_000, bid=0.57, ask=0.60)
    overlay = LowLatencyEntryOverlay(
        queue_path,
        config=LowLatencyOverlayConfig(
            enabled=True,
            max_price_drift_from_signal=0.08,
        ),
    )
    overlay.refresh()

    decision = overlay.evaluate_entry(_signal(market_implied_prob=0.50), now_ms=19_000)

    assert decision.passed is False
    assert decision.reason == "overlay_price_drift_from_signal"
    assert decision.price_drift_from_signal == pytest.approx(0.10)


def test_low_latency_overlay_missing_quote_passes_by_default(tmp_path: Path) -> None:
    overlay = LowLatencyEntryOverlay(
        tmp_path / "missing.jsonl",
        config=LowLatencyOverlayConfig(enabled=True),
    )

    decision = overlay.evaluate_entry(_signal(), now_ms=19_000)

    assert decision.passed is True
    assert decision.reason == "overlay_missing_quote"


def test_low_latency_overlay_missing_quote_can_skip(tmp_path: Path) -> None:
    overlay = LowLatencyEntryOverlay(
        tmp_path / "missing.jsonl",
        config=LowLatencyOverlayConfig(
            enabled=True,
            missing_quote_action="skip",
        ),
    )

    decision = overlay.evaluate_entry(_signal(), now_ms=19_000)

    assert decision.passed is False
    assert decision.reason == "overlay_missing_quote"
