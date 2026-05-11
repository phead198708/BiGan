"""Per-asset stream-outage detector (issue #5).

Tracks the last time we saw a WS message for each subscribed asset_id and
emits :class:`GapEvent` objects when a previously-silent asset's activity
resumes. Two thresholds drive the state machine:

- ``silence_threshold_ms``  — minimum silence duration to declare a gap.
                              Choose a multiple of the busiest asset's
                              expected inter-message interval so micro
                              hiccups don't trip the recovery flow.
- ``min_gap_resume_ms``     — minimum delta between gap_start and the
                              first new message for the gap to be
                              "resolved". Below this, a single late
                              packet won't end an outage prematurely.

The detector is **synchronous and single-threaded**. The runner calls
``note(asset_id, ts_ms)`` from the WS event handler and ``tick(now_ms)``
from a periodic watchdog task; both run inside the same asyncio loop.

Two notification points:

- ``on_gap_started`` — fired exactly once when an asset transitions into
  the silent state (so an operator alert fires immediately, before
  recovery completes).
- ``GapEvent`` returned from ``tick()`` — fired once per gap when the
  asset resumes producing messages; carries gap_start/gap_end/duration
  so the backfill service can pull the right window via REST.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class GapEvent:
    """A resolved silence interval for one asset_id."""

    asset_id: str
    gap_start_ms: int
    gap_end_ms: int

    @property
    def silence_duration_ms(self) -> int:
        return self.gap_end_ms - self.gap_start_ms


@dataclass(slots=True)
class _AssetState:
    """Per-asset bookkeeping for the detector."""

    last_seen_ms: int
    in_gap: bool = False
    gap_start_ms: int = 0
    gap_alerted: bool = False


#: Sentinel: callers passing a known-bogus timestamp (e.g. 0) should be
#: ignored without breaking the detector.
_INVALID_TS: Final[int] = 0


class GapDetector:
    """State machine for detecting and resolving WS stream gaps."""

    def __init__(
        self,
        *,
        silence_threshold_ms: int = 30_000,
        min_gap_resume_ms: int = 1_000,
        on_gap_started: Callable[[str, int], None] | None = None,
    ) -> None:
        if silence_threshold_ms <= 0:
            raise ValueError("silence_threshold_ms must be positive")
        if min_gap_resume_ms < 0:
            raise ValueError("min_gap_resume_ms must be non-negative")
        self._silence_threshold_ms = silence_threshold_ms
        self._min_gap_resume_ms = min_gap_resume_ms
        self._on_gap_started = on_gap_started
        self._states: dict[str, _AssetState] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def note(self, asset_id: str, ts_ms: int) -> GapEvent | None:
        """Record activity for ``asset_id``. Returns a resolved gap if any.

        Called from the WS event handler. ``ts_ms`` is the receive_time
        of the inbound message (not the upstream event timestamp), so a
        delayed-but-just-arrived event still indicates the link is alive.
        """
        if ts_ms <= _INVALID_TS:
            return None

        st = self._states.get(asset_id)
        if st is None:
            self._states[asset_id] = _AssetState(last_seen_ms=ts_ms)
            return None

        gap_event: GapEvent | None = None
        if st.in_gap and ts_ms - st.gap_start_ms >= self._min_gap_resume_ms:
            gap_event = GapEvent(
                asset_id=asset_id,
                gap_start_ms=st.gap_start_ms,
                gap_end_ms=ts_ms,
            )
            logger.info(
                "gap.resolved",
                extra={
                    "asset_id": asset_id,
                    "gap_start_ms": st.gap_start_ms,
                    "gap_end_ms": ts_ms,
                    "silence_duration_ms": gap_event.silence_duration_ms,
                },
            )
            st.in_gap = False
            st.gap_start_ms = 0
            st.gap_alerted = False

        if ts_ms > st.last_seen_ms:
            st.last_seen_ms = ts_ms
        return gap_event

    def tick(self, now_ms: int) -> None:
        """Periodic check: mark any asset that has gone silent past the
        threshold. Does not emit GapEvents — those only come from
        :meth:`note` when activity resumes.
        """
        if now_ms <= _INVALID_TS:
            return
        for asset_id, st in self._states.items():
            if st.in_gap:
                continue
            if now_ms - st.last_seen_ms < self._silence_threshold_ms:
                continue
            st.in_gap = True
            st.gap_start_ms = st.last_seen_ms
            if not st.gap_alerted:
                st.gap_alerted = True
                logger.warning(
                    "gap.detected",
                    extra={
                        "asset_id": asset_id,
                        "last_seen_ms": st.last_seen_ms,
                        "now_ms": now_ms,
                        "silence_duration_ms": now_ms - st.last_seen_ms,
                    },
                )
                if self._on_gap_started is not None:
                    try:
                        self._on_gap_started(asset_id, st.last_seen_ms)
                    except Exception:  # noqa: BLE001
                        logger.exception("gap.callback_failed")

    def forget(self, asset_id: str) -> None:
        """Drop all state for ``asset_id`` (e.g. when it's unsubscribed)."""
        self._states.pop(asset_id, None)

    def reset(self) -> None:
        """Drop all per-asset state (used in tests / hard reconnects)."""
        self._states.clear()

    # ------------------------------------------------------------------
    # Inspection (used by tests + ops tooling)
    # ------------------------------------------------------------------

    def is_in_gap(self, asset_id: str) -> bool:
        st = self._states.get(asset_id)
        return st is not None and st.in_gap

    def last_seen_ms(self, asset_id: str) -> int | None:
        st = self._states.get(asset_id)
        return st.last_seen_ms if st else None

    def tracked_assets(self) -> list[str]:
        return list(self._states.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GapStats:
    """Mutable counter aggregator suitable for prometheus instrumentation."""

    detected: int = 0
    resolved: int = 0
    by_asset: dict[str, int] = field(default_factory=dict)

    def record_detected(self, asset_id: str) -> None:
        self.detected += 1
        self.by_asset[asset_id] = self.by_asset.get(asset_id, 0) + 1

    def record_resolved(self, asset_id: str) -> None:
        self.resolved += 1
