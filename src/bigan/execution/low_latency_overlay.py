"""Low-latency entry overlay for Phase 4 executor gates.

The overlay is intentionally a veto layer. It consumes the raw 5s/10s-capable
queue produced by the live scorer path and blocks entries that look stale,
overpriced, or locally adverse after the model signal was emitted.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bigan.features.low_latency import JsonlRawQueue, RawQueueItem


@dataclass(frozen=True, slots=True)
class LowLatencyOverlayConfig:
    """Runtime knobs for the executor-side low-latency veto."""

    enabled: bool = False
    max_quote_age_seconds: float = 10.0
    window_seconds: float = 10.0
    max_spread: float | None = 0.05
    adverse_velocity_threshold: float | None = 0.04
    max_price_drift_from_signal: float | None = 0.08
    missing_quote_action: str = "pass"
    max_records_per_refresh: int = 20_000

    def __post_init__(self) -> None:
        if self.max_quote_age_seconds <= 0:
            raise ValueError("max_quote_age_seconds must be positive")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if self.max_spread is not None and self.max_spread < 0:
            raise ValueError("max_spread must be non-negative when set")
        if (
            self.adverse_velocity_threshold is not None
            and self.adverse_velocity_threshold < 0
        ):
            raise ValueError("adverse_velocity_threshold must be non-negative when set")
        if (
            self.max_price_drift_from_signal is not None
            and self.max_price_drift_from_signal < 0
        ):
            raise ValueError("max_price_drift_from_signal must be non-negative when set")
        if self.missing_quote_action not in {"pass", "skip"}:
            raise ValueError("missing_quote_action must be pass or skip")
        if self.max_records_per_refresh <= 0:
            raise ValueError("max_records_per_refresh must be positive")


@dataclass(frozen=True, slots=True)
class LowLatencyOverlayRefreshReport:
    """How much raw queue data was consumed by one refresh."""

    rows_read: int
    top_of_book_rows_applied: int
    next_offset: int
    lines_read: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LowLatencyOverlayDecision:
    """Entry overlay decision and the raw evidence behind it."""

    passed: bool
    reason: str
    canonical_symbol: str
    latest_ts: int | None = None
    quote_age_seconds: float | None = None
    latest_bid: float | None = None
    latest_ask: float | None = None
    latest_mid: float | None = None
    latest_spread: float | None = None
    window_seconds: float | None = None
    window_start_ts: int | None = None
    window_start_mid: float | None = None
    mid_velocity: float | None = None
    signal_market_implied_prob: float | None = None
    price_drift_from_signal: float | None = None
    queue_offset: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _QuoteSample:
    ts: int
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid


class LowLatencyEntryOverlay:
    """Incremental top-of-book overlay over the raw low-latency JSONL queue."""

    def __init__(
        self,
        raw_jsonl_path: Path | str,
        *,
        config: LowLatencyOverlayConfig | None = None,
        start: str = "beginning",
    ) -> None:
        if start not in {"beginning", "tail"}:
            raise ValueError("start must be beginning or tail")
        self.path = Path(raw_jsonl_path)
        self.config = config or LowLatencyOverlayConfig(enabled=True)
        self._queue = JsonlRawQueue(self.path)
        self._offset = self.path.stat().st_size if start == "tail" and self.path.exists() else 0
        self._history: dict[str, deque[_QuoteSample]] = {}

    @property
    def offset(self) -> int:
        return self._offset

    def refresh(self) -> LowLatencyOverlayRefreshReport:
        """Consume newly appended raw queue rows."""

        items, next_offset, lines_read = self._queue.read_from_offset(
            self._offset,
            max_records=self.config.max_records_per_refresh,
        )
        applied = 0
        for item in items:
            if self._apply_item(item):
                applied += 1
        self._offset = next_offset
        return LowLatencyOverlayRefreshReport(
            rows_read=len(items),
            top_of_book_rows_applied=applied,
            next_offset=next_offset,
            lines_read=lines_read,
        )

    def evaluate_entry(
        self,
        signal: Mapping[str, Any],
        *,
        now_ms: int,
    ) -> LowLatencyOverlayDecision:
        """Return whether the latest raw queue state allows this entry."""

        if not self.config.enabled:
            return LowLatencyOverlayDecision(
                passed=True,
                reason="overlay_disabled",
                canonical_symbol=str(signal.get("canonical_symbol") or ""),
                queue_offset=self._offset,
            )
        symbol = str(signal.get("canonical_symbol") or "")
        if not symbol:
            return LowLatencyOverlayDecision(
                passed=self.config.missing_quote_action == "pass",
                reason="overlay_missing_canonical_symbol",
                canonical_symbol=symbol,
                queue_offset=self._offset,
            )
        history = self._history.get(symbol)
        if not history:
            return LowLatencyOverlayDecision(
                passed=self.config.missing_quote_action == "pass",
                reason="overlay_missing_quote",
                canonical_symbol=symbol,
                queue_offset=self._offset,
            )

        latest = history[-1]
        quote_age_seconds = max(0.0, (now_ms - latest.ts) / 1000.0)
        window_start = self._window_start_sample(history, latest_ts=latest.ts)
        mid_velocity = None if window_start is latest else latest.mid - window_start.mid
        signal_market = _optional_float(signal.get("market_implied_prob"))
        price_drift = (
            None
            if signal_market is None
            else latest.ask - signal_market
        )

        decision = LowLatencyOverlayDecision(
            passed=True,
            reason="overlay_pass",
            canonical_symbol=symbol,
            latest_ts=latest.ts,
            quote_age_seconds=quote_age_seconds,
            latest_bid=latest.bid,
            latest_ask=latest.ask,
            latest_mid=latest.mid,
            latest_spread=latest.spread,
            window_seconds=self.config.window_seconds,
            window_start_ts=window_start.ts,
            window_start_mid=window_start.mid,
            mid_velocity=mid_velocity,
            signal_market_implied_prob=signal_market,
            price_drift_from_signal=price_drift,
            queue_offset=self._offset,
        )
        skip_reason = self._skip_reason(decision)
        if skip_reason is None:
            return decision
        return LowLatencyOverlayDecision(
            **{**decision.to_dict(), "passed": False, "reason": skip_reason}
        )

    def _apply_item(self, item: RawQueueItem) -> bool:
        if item.table != "raw_top_of_book":
            return False
        row = item.row
        symbol = str(row.get("canonical_symbol") or "")
        if not symbol:
            return False
        bid = _optional_float(row.get("bid_price"))
        ask = _optional_float(row.get("ask_price"))
        if bid is None or ask is None or bid < 0 or ask <= 0 or ask < bid:
            return False
        ts = _row_ts_ms(row, item.published_at_ms)
        sample = _QuoteSample(ts=ts, bid=bid, ask=ask)
        history = self._history.setdefault(symbol, deque())
        history.append(sample)
        self._prune_history(history, latest_ts=ts)
        return True

    def _prune_history(self, history: deque[_QuoteSample], *, latest_ts: int) -> None:
        retain_ms = int(
            max(self.config.window_seconds, self.config.max_quote_age_seconds) * 1000
        ) + 5_000
        cutoff = latest_ts - retain_ms
        while len(history) > 1 and history[0].ts < cutoff:
            history.popleft()

    def _window_start_sample(
        self,
        history: deque[_QuoteSample],
        *,
        latest_ts: int,
    ) -> _QuoteSample:
        cutoff = latest_ts - int(self.config.window_seconds * 1000)
        candidate = history[0]
        for sample in history:
            if sample.ts <= cutoff:
                candidate = sample
                continue
            break
        if candidate.ts <= cutoff or len(history) == 1:
            return candidate
        return history[0]

    def _skip_reason(self, decision: LowLatencyOverlayDecision) -> str | None:
        if (
            decision.quote_age_seconds is not None
            and decision.quote_age_seconds > self.config.max_quote_age_seconds
        ):
            return "overlay_quote_stale"
        if (
            self.config.max_spread is not None
            and decision.latest_spread is not None
            and decision.latest_spread > self.config.max_spread
        ):
            return "overlay_spread_too_wide"
        if (
            self.config.max_price_drift_from_signal is not None
            and decision.price_drift_from_signal is not None
            and decision.price_drift_from_signal > self.config.max_price_drift_from_signal
        ):
            return "overlay_price_drift_from_signal"
        if (
            self.config.adverse_velocity_threshold is not None
            and decision.mid_velocity is not None
            and decision.mid_velocity > self.config.adverse_velocity_threshold
        ):
            return "overlay_adverse_side_velocity"
        return None


def _row_ts_ms(row: Mapping[str, Any], fallback: int) -> int:
    for key in ("ts", "message_ts", "capture_timestamp_ms", "ingest_ts"):
        value = row.get(key)
        parsed = _optional_float(value)
        if parsed is not None:
            return int(parsed)
    return int(fallback)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
