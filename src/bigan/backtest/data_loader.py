"""Load historical CLOB top-of-book snapshots from Parquet or CSV."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from bigan.data.polymarket_clob import MarketSnapshot

SNAPSHOT_COLUMNS: tuple[str, ...] = (
    "timestamp_ms",
    "window_id",
    "yes_bid",
    "yes_ask",
    "no_bid",
    "no_ask",
    "last_traded_price",
    "yes_bid_size",
    "yes_ask_size",
    "no_bid_size",
    "no_ask_size",
)
SPOT_COLUMN = "spot_price"
_TIMESTAMP_ALIASES = ("timestamp_ms", "timestamp", "ts_ms", "ts")


@dataclass(frozen=True, slots=True)
class LoadedClob:
    """Parsed CLOB tape plus optional per-tick BTC spot."""

    snapshots: tuple[MarketSnapshot, ...]
    spot_prices: tuple[float, ...] | None
    dropped_stale: int


def load_clob_snapshots(path: str | Path) -> LoadedClob:
    """Load a Parquet or CSV CLOB log into ``MarketSnapshot`` tuples.

    Required columns: ``timestamp_ms`` (aliases ``timestamp`` / ``ts_ms`` /
    ``ts``), ``window_id``, ``yes_bid``, ``yes_ask``, ``no_bid``, ``no_ask``.
    ``last_traded_price`` and ``spot_price`` are optional.
    """

    source = Path(path)
    table = _read_table(source)
    return snapshots_from_table(table)


def write_clob_snapshots(
    path: str | Path,
    snapshots: Sequence[MarketSnapshot],
    *,
    spot_prices: Sequence[float] | None = None,
) -> None:
    """Write snapshots to Parquet (``.parquet`` / ``.pq``) or CSV."""

    if spot_prices is not None and len(spot_prices) != len(snapshots):
        raise ValueError("spot_prices must match snapshots length")
    table = _snapshots_to_table(snapshots, spot_prices=spot_prices)
    destination = Path(path)
    suffix = destination.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        pq.write_table(table, destination)
        return
    if suffix == ".csv":
        pacsv.write_csv(table, destination)
        return
    raise ValueError(f"unsupported CLOB snapshot suffix: {destination.suffix}")


def snapshots_from_table(table: pa.Table) -> LoadedClob:
    """Convert an Arrow table with CLOB columns into snapshots."""

    names = {str(name): i for i, name in enumerate(table.column_names)}
    ts_col = _resolve_timestamp_column(names)
    window_col = _require_column(names, "window_id")
    yes_bid_col = _require_column(names, "yes_bid")
    yes_ask_col = _require_column(names, "yes_ask")
    no_bid_col = _require_column(names, "no_bid")
    no_ask_col = _require_column(names, "no_ask")
    last_idx = names.get("last_traded_price")
    size_indices = {
        column: names.get(column)
        for column in ("yes_bid_size", "yes_ask_size", "no_bid_size", "no_ask_size")
    }
    spot_idx = names.get(SPOT_COLUMN)

    timestamps = table.column(ts_col).to_pylist()
    windows = table.column(window_col).to_pylist()
    yes_bids = table.column(yes_bid_col).to_pylist()
    yes_asks = table.column(yes_ask_col).to_pylist()
    no_bids = table.column(no_bid_col).to_pylist()
    no_asks = table.column(no_ask_col).to_pylist()
    lasts = table.column(last_idx).to_pylist() if last_idx is not None else None
    sizes = {
        column: table.column(index).to_pylist() if index is not None else None
        for column, index in size_indices.items()
    }
    spots_raw = table.column(spot_idx).to_pylist() if spot_idx is not None else None

    snapshots: list[MarketSnapshot] = []
    spots: list[float] | None = [] if spots_raw is not None else None
    dropped_stale = 0
    last_ts: int | None = None
    n = table.num_rows
    for i in range(n):
        try:
            ts_ms = int(timestamps[i])
            window_id = str(windows[i])
            yes_bid = _finite_price("yes_bid", yes_bids[i])
            yes_ask = _finite_price("yes_ask", yes_asks[i])
            no_bid = _finite_price("no_bid", no_bids[i])
            no_ask = _finite_price("no_ask", no_asks[i])
            last_px = 0.0 if lasts is None else _optional_price(lasts[i])
            quote_sizes = {
                column: (
                    0.0
                    if values is None
                    else _finite_size(column, values[i])
                )
                for column, values in sizes.items()
            }
            spot = (
                None
                if spots_raw is None
                else _finite_price(SPOT_COLUMN, spots_raw[i])
            )
        except (TypeError, ValueError):
            dropped_stale += 1
            continue
        if not window_id.strip():
            dropped_stale += 1
            continue
        if yes_bid > yes_ask or no_bid > no_ask:
            dropped_stale += 1
            continue
        if last_ts is not None and ts_ms < last_ts:
            dropped_stale += 1
            continue
        snapshot = MarketSnapshot(
            timestamp_ms=ts_ms,
            window_id=window_id,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
            last_traded_price=last_px,
            **quote_sizes,
        )
        last_ts = ts_ms
        snapshots.append(snapshot)
        if spots is not None and spot is not None:
            spots.append(spot)
    return LoadedClob(
        snapshots=tuple(snapshots),
        spot_prices=tuple(spots) if spots is not None else None,
        dropped_stale=dropped_stale,
    )


def generate_synthetic_clob(
    *,
    n_ticks: int = 128,
    window_id: str = "btc-updown-15m-bt",
    start_ts_ms: int = 100_000,
    step_ms: int = 200,
    yes_ask: float = 0.40,
    no_ask: float = 0.90,
    start_yes_bid: float = 0.30,
    lift: float = 0.002,
    noise: float = 0.0,
    seed: int | None = None,
) -> tuple[MarketSnapshot, ...]:
    """Build a synthetic YES/NO tape with optional trend and Gaussian noise."""

    if n_ticks < 1:
        raise ValueError("n_ticks must be positive")
    if step_ms <= 0:
        raise ValueError("step_ms must be positive")
    rng = random.Random(seed)
    snapshots: list[MarketSnapshot] = []
    bid = float(start_yes_bid)
    ask = float(yes_ask)
    for i in range(n_ticks):
        in_burst = i >= n_ticks - 8
        should_lift = in_burst or i % 3 == 0
        if should_lift and i > 0:
            bid += float(lift)
        if noise > 0.0:
            bid = min(ask - 1e-4, max(0.01, bid + rng.gauss(0.0, noise)))
        if bid >= ask:
            bid = ask - 1e-4
        no_bid = min(0.99, max(0.01, 1.0 - ask - 0.02))
        snapshots.append(
            MarketSnapshot(
                timestamp_ms=int(start_ts_ms + i * step_ms),
                window_id=str(window_id),
                yes_bid=float(bid),
                yes_ask=ask,
                no_bid=float(no_bid),
                no_ask=float(no_ask),
                last_traded_price=float(bid),
                yes_bid_size=10_000.0,
                yes_ask_size=10_000.0,
                no_bid_size=10_000.0,
                no_ask_size=10_000.0,
            )
        )
    return tuple(snapshots)


def _read_table(path: Path) -> pa.Table:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pq.read_table(path)
    if suffix == ".csv":
        return pacsv.read_csv(path)
    raise ValueError(f"unsupported CLOB snapshot suffix: {path.suffix}")


def _snapshots_to_table(
    snapshots: Sequence[MarketSnapshot],
    *,
    spot_prices: Sequence[float] | None,
) -> pa.Table:
    payload: dict[str, object] = {
        "timestamp_ms": [int(row.timestamp_ms) for row in snapshots],
        "window_id": [str(row.window_id) for row in snapshots],
        "yes_bid": [float(row.yes_bid) for row in snapshots],
        "yes_ask": [float(row.yes_ask) for row in snapshots],
        "no_bid": [float(row.no_bid) for row in snapshots],
        "no_ask": [float(row.no_ask) for row in snapshots],
        "last_traded_price": [float(row.last_traded_price) for row in snapshots],
        "yes_bid_size": [float(row.yes_bid_size) for row in snapshots],
        "yes_ask_size": [float(row.yes_ask_size) for row in snapshots],
        "no_bid_size": [float(row.no_bid_size) for row in snapshots],
        "no_ask_size": [float(row.no_ask_size) for row in snapshots],
    }
    if spot_prices is not None:
        payload[SPOT_COLUMN] = [float(px) for px in spot_prices]
    return pa.table(payload)


def _resolve_timestamp_column(names: Mapping[str, int]) -> int:
    for alias in _TIMESTAMP_ALIASES:
        if alias in names:
            return names[alias]
    raise ValueError("CLOB table is missing a timestamp column")


def _require_column(names: Mapping[str, int], column: str) -> int:
    if column not in names:
        raise ValueError(f"CLOB table is missing column '{column}'")
    return names[column]


def _finite_price(name: str, value: object) -> float:
    out = float(value)  # type: ignore[arg-type]
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be a positive finite price")
    return out


def _optional_price(value: object) -> float:
    if value is None:
        return 0.0
    out = float(value)  # type: ignore[arg-type]
    if not math.isfinite(out):
        return 0.0
    return out


def _finite_size(name: str, value: object) -> float:
    out = float(value)  # type: ignore[arg-type]
    if not math.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be a non-negative finite size")
    return out
