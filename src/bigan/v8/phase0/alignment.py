"""Strict point-in-time alignment primitives."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from bigan.v8.phase0.contracts import FeatureVector, Label, MarketData


@dataclass(frozen=True, slots=True)
class AlignedMarketSeries:
    """Sorted market data grouped by ``(source, instrument_id)``."""

    rows: tuple[MarketData, ...]
    groups: dict[tuple[str, str], tuple[MarketData, ...]]
    ts_index: dict[tuple[str, str], tuple[int, ...]]

    def latest_at(self, source: str, instrument_id: str, decision_ts: int) -> MarketData | None:
        group_key = (source, instrument_id)
        rows = self.groups.get(group_key, ())
        timestamps = self.ts_index.get(group_key, ())
        idx = bisect_right(timestamps, decision_ts) - 1
        while idx >= 0:
            row = rows[idx]
            if row.ts <= decision_ts and int(row.available_at_ts or row.ts) <= decision_ts:
                return row
            idx -= 1
        return None

    def first_at_or_after(
        self,
        source: str,
        instrument_id: str,
        target_ts: int,
    ) -> MarketData | None:
        group_key = (source, instrument_id)
        rows = self.groups.get(group_key, ())
        timestamps = self.ts_index.get(group_key, ())
        idx = bisect_left(timestamps, target_ts)
        if idx >= len(rows):
            return None
        return rows[idx]

    def window(
        self,
        source: str,
        instrument_id: str,
        *,
        start_exclusive: int,
        end_inclusive: int,
    ) -> tuple[MarketData, ...]:
        group_key = (source, instrument_id)
        rows = self.groups.get(group_key, ())
        timestamps = self.ts_index.get(group_key, ())
        start_idx = bisect_right(timestamps, start_exclusive)
        end_idx = bisect_right(timestamps, end_inclusive)
        return tuple(
            row
            for row in rows[start_idx:end_idx]
            if row.ts <= end_inclusive and int(row.available_at_ts or row.ts) <= end_inclusive
        )

    def decision_times(
        self,
        source: str,
        instrument_id: str,
        *,
        frequency_ms: int | None = None,
        since_ts: int | None = None,
        until_ts: int | None = None,
    ) -> tuple[int, ...]:
        timestamps = self.ts_index.get((source, instrument_id), ())
        if not timestamps:
            return ()
        start = timestamps[0] if since_ts is None else max(timestamps[0], since_ts)
        end = timestamps[-1] if until_ts is None else min(timestamps[-1], until_ts)
        if start > end:
            return ()
        if frequency_ms is None:
            return tuple(ts for ts in timestamps if start <= ts <= end)
        if frequency_ms <= 0:
            raise ValueError("frequency_ms must be positive")
        first = _ceil_to_grid(start, frequency_ms)
        return tuple(range(first, end + 1, frequency_ms))


class TimeAlignmentEngine:
    """Build aligned series and enforce feature/label timestamp contracts."""

    def align_market_data(self, rows: Iterable[MarketData]) -> AlignedMarketSeries:
        sorted_rows = tuple(
            sorted(
                rows,
                key=lambda row: (
                    row.source,
                    row.instrument_id,
                    row.ts,
                    row.available_at_ts or row.ts,
                    row.sequence if row.sequence is not None else -1,
                ),
            )
        )
        groups_mut: dict[tuple[str, str], list[MarketData]] = defaultdict(list)
        seen: set[tuple[str, str, int, int | None]] = set()
        for row in sorted_rows:
            key = (row.source, row.instrument_id, row.ts, row.sequence)
            if key in seen:
                raise ValueError(
                    "duplicate aligned market row for "
                    f"{row.source}/{row.instrument_id} at {row.ts}"
                )
            seen.add(key)
            if int(row.available_at_ts or row.ts) < row.ts:
                raise ValueError("market row available_at_ts cannot precede ts")
            groups_mut[(row.source, row.instrument_id)].append(row)

        groups = {key: tuple(value) for key, value in sorted(groups_mut.items())}
        ts_index = {key: tuple(row.ts for row in value) for key, value in groups.items()}
        for key, timestamps in ts_index.items():
            if tuple(sorted(timestamps)) != timestamps:
                raise ValueError(f"timestamps are not monotonic for {key}")
        return AlignedMarketSeries(rows=sorted_rows, groups=groups, ts_index=ts_index)

    def enforce_feature_causality(self, features: Iterable[FeatureVector]) -> None:
        offenders = [
            feature
            for feature in features
            if feature.max_input_ts > feature.decision_ts
            or feature.feature_cutoff_ts > feature.decision_ts
            or any(
                provenance.input_end_ts > feature.decision_ts
                or provenance.available_at_ts > feature.decision_ts
                for provenance in feature.provenance.values()
            )
        ]
        if offenders:
            first = offenders[0]
            raise ValueError(
                "feature causality violation at "
                f"{first.source}/{first.instrument_id}/{first.decision_ts}"
            )

    def enforce_label_after_feature(self, labels: Iterable[Label]) -> None:
        offenders = [
            label
            for label in labels
            if label.label_ts < label.decision_ts + label.horizon_ms
        ]
        if offenders:
            first = offenders[0]
            raise ValueError(
                "label timestamp violates horizon at "
                f"{first.source}/{first.instrument_id}/{first.decision_ts}"
            )


def _ceil_to_grid(ts: int, grid_ms: int) -> int:
    return ((ts + grid_ms - 1) // grid_ms) * grid_ms

