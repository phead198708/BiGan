"""Unit tests for the per-asset gap detector (issue #5)."""

from __future__ import annotations

from bigan.ingestion.gap_detector import GapDetector, GapEvent


def _ms(seconds: float) -> int:
    return int(seconds * 1000)


# ---------------------------------------------------------------------------
# Basic state transitions
# ---------------------------------------------------------------------------


def test_first_note_does_not_emit_gap() -> None:
    d = GapDetector(silence_threshold_ms=_ms(10), min_gap_resume_ms=_ms(0.5))
    assert d.note("a", _ms(1.0)) is None
    assert d.last_seen_ms("a") == _ms(1.0)
    assert not d.is_in_gap("a")


def test_continuous_activity_does_not_enter_gap() -> None:
    d = GapDetector(silence_threshold_ms=_ms(5), min_gap_resume_ms=_ms(0.5))
    for s in range(0, 20):
        d.note("a", _ms(s))
        d.tick(_ms(s))
    assert not d.is_in_gap("a")


def test_silence_past_threshold_marks_in_gap() -> None:
    d = GapDetector(silence_threshold_ms=_ms(5), min_gap_resume_ms=_ms(0.5))
    d.note("a", _ms(1.0))
    d.tick(_ms(1.5))  # 0.5s silence - not yet a gap
    assert not d.is_in_gap("a")
    d.tick(_ms(7.0))  # 6s silence - over threshold
    assert d.is_in_gap("a")


def test_gap_resolves_on_resume() -> None:
    d = GapDetector(silence_threshold_ms=_ms(5), min_gap_resume_ms=_ms(0.5))
    d.note("a", _ms(1.0))
    d.tick(_ms(8.0))  # mark as in_gap; gap_start = last_seen = 1.0
    assert d.is_in_gap("a")
    resolved = d.note("a", _ms(8.5))
    assert isinstance(resolved, GapEvent)
    assert resolved.asset_id == "a"
    assert resolved.gap_start_ms == _ms(1.0)
    assert resolved.gap_end_ms == _ms(8.5)
    assert resolved.silence_duration_ms == _ms(7.5)
    assert not d.is_in_gap("a")


def test_resume_below_min_resume_threshold_is_ignored() -> None:
    """A single late packet should not end a gap if it arrives within
    ``min_gap_resume_ms`` of the gap's start (i.e. would have been a
    legitimate inter-message lull)."""
    d = GapDetector(silence_threshold_ms=_ms(5), min_gap_resume_ms=_ms(2))
    d.note("a", _ms(1.0))
    d.tick(_ms(8.0))
    # Resume packet sits at gap_start + 1ms, less than 2s threshold.
    resolved = d.note("a", _ms(1.0) + 1)
    assert resolved is None
    assert d.is_in_gap("a")  # still in gap
    # A real resume eventually arrives.
    resolved = d.note("a", _ms(10.0))
    assert resolved is not None
    assert not d.is_in_gap("a")


# ---------------------------------------------------------------------------
# Multi-asset isolation
# ---------------------------------------------------------------------------


def test_assets_track_independently() -> None:
    d = GapDetector(silence_threshold_ms=_ms(5), min_gap_resume_ms=_ms(0.5))
    d.note("a", _ms(1.0))
    d.note("b", _ms(1.0))
    # Only "a" goes silent.
    d.note("b", _ms(8.0))
    d.tick(_ms(8.0))
    assert d.is_in_gap("a")
    assert not d.is_in_gap("b")


# ---------------------------------------------------------------------------
# Detection callback
# ---------------------------------------------------------------------------


def test_on_gap_started_callback_fires_once() -> None:
    calls: list[tuple[str, int]] = []

    def cb(asset_id: str, last_seen_ms: int) -> None:
        calls.append((asset_id, last_seen_ms))

    d = GapDetector(silence_threshold_ms=_ms(5), on_gap_started=cb)
    d.note("a", _ms(1.0))
    d.tick(_ms(8.0))
    d.tick(_ms(9.0))  # still silent — must not fire again
    assert calls == [("a", _ms(1.0))]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_invalid_timestamps_are_ignored() -> None:
    d = GapDetector(silence_threshold_ms=_ms(5))
    assert d.note("a", 0) is None
    assert d.note("a", -1) is None
    assert d.last_seen_ms("a") is None


def test_forget_drops_state() -> None:
    d = GapDetector(silence_threshold_ms=_ms(5))
    d.note("a", _ms(1.0))
    assert "a" in d.tracked_assets()
    d.forget("a")
    assert d.last_seen_ms("a") is None


def test_constructor_validates_arguments() -> None:
    import pytest

    with pytest.raises(ValueError):
        GapDetector(silence_threshold_ms=0)
    with pytest.raises(ValueError):
        GapDetector(silence_threshold_ms=1, min_gap_resume_ms=-1)
