"""Binance top-of-book Order Flow Imbalance (OFI) engine.

Computes Cont-style best-level event imbalance from millisecond bid/ask
updates, then emits an EMA-smoothed, windowed Z-score clipped against spoofing.
The class is a pure ingest sink: a live ``bookTicker`` / depth stream can push
payloads in; unit tests inject the same top-of-book tuples.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

DEFAULT_EMA_ALPHA = 0.2
DEFAULT_WINDOW_MS = 60_000
DEFAULT_MAX_EVENTS_CAP = 100_000
MIN_ZSCORE_SAMPLES = 20
ZSCORE_CLIP = 3.0
RECALIBRATE_EVERY = 5_000
_VARIANCE_FLOOR = 1e-18


def cont_bid_imbalance(
    prev_price: float,
    prev_qty: float,
    price: float,
    qty: float,
) -> float:
    """Best-bid contribution ``I_b(t)`` from one top-of-book transition."""

    if price > prev_price:
        return qty
    if price < prev_price:
        return -prev_qty
    return qty - prev_qty


def cont_ask_imbalance(
    prev_price: float,
    prev_qty: float,
    price: float,
    qty: float,
) -> float:
    """Best-ask contribution ``I_a(t)`` from one top-of-book transition."""

    if price < prev_price:
        return qty
    if price > prev_price:
        return -prev_qty
    return qty - prev_qty


@dataclass(frozen=True, slots=True)
class TopOfBook:
    """One Binance best bid/ask observation."""

    ts_ms: int
    bid_price: float
    bid_qty: float
    ask_price: float
    ask_qty: float


@dataclass(frozen=True, slots=True)
class OFISnapshot:
    """One computed OFI event after a depth update that had a prior book."""

    ts_ms: int
    bid_price: float
    bid_qty: float
    ask_price: float
    ask_qty: float
    i_b: float
    i_a: float
    raw_ofi: float
    ema_ofi: float
    z_ofi: float


class BinanceOFICalculator:
    """Millisecond L2 top-of-book OFI with EMA smoothing and clipped Z-score.

    Parameters
    ----------
    ema_alpha:
        EMA weight on the newest raw OFI. Must be in ``(0, 1]``.
    window_ms:
        Trailing time window that holds EMA samples for Z-score.
    zscore_min_samples:
        Below this count ``get_normalized_ofi()`` returns ``0.0``.
    zscore_clip:
        Absolute cap applied to the Z-score (spoofing defense).
    max_events_cap:
        Hard memory bound on retained EMA samples. Time-window eviction is
        independent of this cap and uses only ``ts_ms - window_ms``.
    symbol:
        Expected Binance instrument id for payload ingest.
    """

    __slots__ = (
        "ema_alpha",
        "window_ms",
        "zscore_min_samples",
        "zscore_clip",
        "max_events_cap",
        "symbol",
        "_prev",
        "_ema",
        "_last_raw",
        "_last_z",
        "_last_ts_ms",
        "_last_update_id",
        "_samples",
        "_sum",
        "_sum_sq",
        "_recalc_counter",
    )

    def __init__(
        self,
        *,
        ema_alpha: float = DEFAULT_EMA_ALPHA,
        window_ms: int = DEFAULT_WINDOW_MS,
        zscore_min_samples: int = MIN_ZSCORE_SAMPLES,
        zscore_clip: float = ZSCORE_CLIP,
        max_events_cap: int = DEFAULT_MAX_EVENTS_CAP,
        symbol: str = "BTCUSDT",
    ) -> None:
        if not 0.0 < float(ema_alpha) <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if int(window_ms) <= 0:
            raise ValueError("window_ms must be positive")
        if int(zscore_min_samples) < 1:
            raise ValueError("zscore_min_samples must be positive")
        if float(zscore_clip) <= 0.0:
            raise ValueError("zscore_clip must be positive")
        if int(max_events_cap) < 1:
            raise ValueError("max_events_cap must be positive")
        if not str(symbol).strip():
            raise ValueError("symbol must be non-empty")
        self.ema_alpha = float(ema_alpha)
        self.window_ms = int(window_ms)
        self.zscore_min_samples = int(zscore_min_samples)
        self.zscore_clip = float(zscore_clip)
        self.max_events_cap = int(max_events_cap)
        self.symbol = str(symbol).upper()
        self._prev: TopOfBook | None = None
        self._ema: float | None = None
        self._last_raw = 0.0
        self._last_z = 0.0
        self._last_ts_ms: int | None = None
        self._last_update_id: int | None = None
        self._samples: deque[tuple[int, float]] = deque()
        self._sum = 0.0
        self._sum_sq = 0.0
        self._recalc_counter = 0

    @property
    def last_raw_ofi(self) -> float:
        return self._last_raw

    @property
    def last_ema_ofi(self) -> float:
        return 0.0 if self._ema is None else self._ema

    @property
    def last_timestamp_ms(self) -> int | None:
        """Timestamp of the latest accepted top-of-book observation."""

        return self._last_ts_ms

    @property
    def last_update_id(self) -> int | None:
        """Binance ``bookTicker`` update id of the latest accepted event."""

        return self._last_update_id

    def get_normalized_ofi(self) -> float:
        """Return the latest clipped Z-score, or 0.0 until the window is ready."""

        if len(self._samples) < self.zscore_min_samples:
            return 0.0
        return self._last_z

    def on_depth_update(
        self,
        *,
        bid_price: float,
        bid_qty: float,
        ask_price: float,
        ask_qty: float,
        ts_ms: int,
    ) -> OFISnapshot | None:
        """Ingest one best-level depth update.

        The first observation only seeds state and returns ``None``. Later
        updates emit Cont ``I_b``, ``I_a``, ``raw_ofi = I_b - I_a``, EMA, and
        the clipped Z-score.
        """

        book = _validated_top_of_book(
            ts_ms=ts_ms,
            bid_price=bid_price,
            bid_qty=bid_qty,
            ask_price=ask_price,
            ask_qty=ask_qty,
        )
        applied = self._apply_depth(book)
        if applied is None:
            return None
        i_b, i_a, raw_ofi, ema_ofi, z_ofi = applied
        return OFISnapshot(
            ts_ms=book.ts_ms,
            bid_price=book.bid_price,
            bid_qty=book.bid_qty,
            ask_price=book.ask_price,
            ask_qty=book.ask_qty,
            i_b=i_b,
            i_a=i_a,
            raw_ofi=raw_ofi,
            ema_ofi=ema_ofi,
            z_ofi=z_ofi,
        )

    def update_and_get_z(
        self,
        *,
        bid_price: float,
        bid_qty: float,
        ask_price: float,
        ask_qty: float,
        ts_ms: int,
    ) -> float:
        """Ingest one depth update and return only the clipped Z-score.

        Intended for high-frequency strategy loops that do not need an
        ``OFISnapshot``. The first observation seeds state and returns ``0.0``.
        """

        book = _validated_top_of_book(
            ts_ms=ts_ms,
            bid_price=bid_price,
            bid_qty=bid_qty,
            ask_price=ask_price,
            ask_qty=ask_qty,
        )
        applied = self._apply_depth(book)
        if applied is None:
            return 0.0
        return applied[4]

    def _apply_depth(self, book: TopOfBook) -> tuple[float, float, float, float, float] | None:
        prev = self._prev
        if prev is None:
            self._prev = book
            self._last_ts_ms = book.ts_ms
            return None
        if book.ts_ms < prev.ts_ms:
            return None
        if book.ts_ms == prev.ts_ms and book == prev:
            return None
        i_b = cont_bid_imbalance(prev.bid_price, prev.bid_qty, book.bid_price, book.bid_qty)
        i_a = cont_ask_imbalance(prev.ask_price, prev.ask_qty, book.ask_price, book.ask_qty)
        raw_ofi = i_b - i_a
        if self._ema is None:
            ema_ofi = raw_ofi
        else:
            ema_ofi = self.ema_alpha * raw_ofi + (1.0 - self.ema_alpha) * self._ema
        if not math.isfinite(ema_ofi):
            raise ValueError("EMA produced a non-finite OFI")
        self._prev = book
        self._last_ts_ms = book.ts_ms
        self._ema = ema_ofi
        self._last_raw = raw_ofi
        self._push_sample(book.ts_ms, ema_ofi)
        z_ofi = self._zscore()
        self._last_z = z_ofi
        return i_b, i_a, raw_ofi, ema_ofi, z_ofi

    def on_book_ticker(
        self,
        payload: Mapping[str, object],
        *,
        ts_ms: int | None = None,
    ) -> OFISnapshot | None:
        """Receive a Binance ``bookTicker`` (WS or REST) best-bid/ask payload."""

        wrapped = payload.get("data")
        ticker = wrapped if isinstance(wrapped, Mapping) else payload
        symbol = str(ticker.get("s") or ticker.get("symbol") or self.symbol).upper()
        if symbol != self.symbol:
            raise ValueError(f"unexpected bookTicker symbol {symbol}")
        bid_price = _ticker_float(ticker, "b", "bidPrice")
        bid_qty = _ticker_float(ticker, "B", "bidQty")
        ask_price = _ticker_float(ticker, "a", "askPrice")
        ask_qty = _ticker_float(ticker, "A", "askQty")
        update_id = _optional_int(ticker.get("u"))
        if (
            update_id is not None
            and self._last_update_id is not None
            and update_id <= self._last_update_id
        ):
            return None
        event_ts = _optional_int(ticker.get("E"))
        update_ts = int(ts_ms) if ts_ms is not None else event_ts
        if update_ts is None:
            update_ts = time.time_ns() // 1_000_000
        if self._last_ts_ms is not None and update_ts < self._last_ts_ms:
            return None
        result = self.on_depth_update(
            bid_price=bid_price,
            bid_qty=bid_qty,
            ask_price=ask_price,
            ask_qty=ask_qty,
            ts_ms=update_ts,
        )
        if update_id is not None and (
            self._last_update_id is None or update_id > self._last_update_id
        ):
            self._last_update_id = update_id
        return result

    def on_partial_depth(self, payload: Mapping[str, object], *, ts_ms: int) -> OFISnapshot | None:
        """Receive a Binance partial depth snapshot and use the best bid/ask."""

        bids = payload.get("bids")
        asks = payload.get("asks")
        if not isinstance(bids, (list, tuple)) or not isinstance(asks, (list, tuple)):
            raise ValueError("partial depth payload must include bids and asks")
        if not bids or not asks:
            raise ValueError("partial depth payload is missing the best level")
        best_bid = bids[0]
        best_ask = asks[0]
        return self.on_depth_update(
            bid_price=float(best_bid[0]),
            bid_qty=float(best_bid[1]),
            ask_price=float(best_ask[0]),
            ask_qty=float(best_ask[1]),
            ts_ms=int(ts_ms),
        )

    def reset(self) -> None:
        """Drop all book and window state."""

        self._prev = None
        self._ema = None
        self._last_raw = 0.0
        self._last_z = 0.0
        self._last_ts_ms = None
        self._last_update_id = None
        self._samples.clear()
        self._sum = 0.0
        self._sum_sq = 0.0
        self._recalc_counter = 0

    def _push_sample(self, ts_ms: int, value: float) -> None:
        samples = self._samples
        samples.append((ts_ms, value))
        self._sum += value
        self._sum_sq += value * value
        cutoff = ts_ms - self.window_ms
        while samples and samples[0][0] < cutoff:
            self._evict_oldest()
        while len(samples) > self.max_events_cap:
            self._evict_oldest()
        self._recalc_counter += 1
        if self._recalc_counter >= RECALIBRATE_EVERY:
            self._recalibrate()

    def _evict_oldest(self) -> None:
        _, old = self._samples.popleft()
        self._sum -= old
        self._sum_sq -= old * old

    def _recalibrate(self) -> None:
        total = 0.0
        total_sq = 0.0
        for _, value in self._samples:
            total += value
            total_sq += value * value
        self._sum = total
        self._sum_sq = total_sq
        self._recalc_counter = 0

    def _zscore(self) -> float:
        n = len(self._samples)
        if n < self.zscore_min_samples:
            return 0.0
        mean = self._sum / n
        variance = self._sum_sq / n - mean * mean
        if variance <= _VARIANCE_FLOOR or not math.isfinite(variance):
            return 0.0
        current = self._samples[-1][1]
        z_ofi = (current - mean) / math.sqrt(variance)
        if not math.isfinite(z_ofi):
            return 0.0
        clip = self.zscore_clip
        if z_ofi > clip:
            return clip
        if z_ofi < -clip:
            return -clip
        return z_ofi


def _validated_top_of_book(
    *,
    ts_ms: int,
    bid_price: float,
    bid_qty: float,
    ask_price: float,
    ask_qty: float,
) -> TopOfBook:
    ts = int(ts_ms)
    bid_px = float(bid_price)
    bid_q = float(bid_qty)
    ask_px = float(ask_price)
    ask_q = float(ask_qty)
    values = (bid_px, bid_q, ask_px, ask_q)
    if ts < 0:
        raise ValueError("ts_ms must be non-negative")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("top-of-book prices and quantities must be finite")
    if bid_px <= 0.0 or ask_px <= 0.0:
        raise ValueError("top-of-book prices must be positive")
    if bid_q < 0.0 or ask_q < 0.0:
        raise ValueError("top-of-book quantities must be non-negative")
    if bid_px > ask_px:
        raise ValueError("crossed top of book is not allowed")
    return TopOfBook(
        ts_ms=ts,
        bid_price=bid_px,
        bid_qty=bid_q,
        ask_price=ask_px,
        ask_qty=ask_q,
    )


def _ticker_float(payload: Mapping[str, object], short_key: str, long_key: str) -> float:
    raw = payload.get(short_key)
    if raw is None:
        raw = payload.get(long_key)
    if raw is None:
        raise ValueError(f"Ticker payload missing both '{short_key}' and '{long_key}'")
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Ticker field '{short_key}' must be numeric") from exc


def _optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return None
