"""Minute-grain feature aggregation for ``features_15m_v1``.

The aggregation timestamp is the minute boundary ``feature_ts``. Every lookup
is backward-looking: point-in-time values use the latest input row with
``ts <= feature_ts`` and rolling windows use ``(feature_ts - window, feature_ts]``.
"""

from __future__ import annotations

import math
import time
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from bigan.canonical.query import open_warehouse
from bigan.canonical.writer import WarehouseWriter

from .quality import DEFAULT_QUALITY_CONFIG, FeatureQualityConfig, compute_quality_fields
from .registry import FEATURE_VERSION

BUCKET_MS = 60_000
NEUTRAL_AGGRESSOR_BUY_RATIO = 0.5
NO_TRADE_AVG_SIZE = 0.0
NEUTRAL_RETURN = 0.0
NEUTRAL_ORDERBOOK_IMBALANCE = 0.0


@dataclass(slots=True)
class FeatureBatchReport:
    """Summary of one ``features_15m_v1`` batch."""

    rows_generated: int = 0
    rows_written: int = 0
    feature_version: str = FEATURE_VERSION


@dataclass(slots=True)
class _Series:
    rows: list[dict[str, Any]]
    ts_values: list[int] = field(init=False)

    def __post_init__(self) -> None:
        self.rows.sort(key=lambda row: int(row["ts"]))
        self.ts_values = [int(row["ts"]) for row in self.rows]

    def latest_at(self, ts: int) -> dict[str, Any] | None:
        idx = bisect_right(self.ts_values, ts) - 1
        return self.rows[idx] if idx >= 0 else None

    def window(self, start_exclusive: int, end_inclusive: int) -> list[dict[str, Any]]:
        start = bisect_right(self.ts_values, start_exclusive)
        end = bisect_right(self.ts_values, end_inclusive)
        return self.rows[start:end]


def aggregate_features_15m_v1(
    *,
    top_of_book_rows: Iterable[Mapping[str, Any]],
    orderbook_rows: Iterable[Mapping[str, Any]],
    trade_rows: Iterable[Mapping[str, Any]],
    ingest_ts: int | None = None,
    quality_config: FeatureQualityConfig = DEFAULT_QUALITY_CONFIG,
    since_ms: int | None = None,
    until_ms: int | None = None,
    bucket_ms: int = BUCKET_MS,
) -> list[dict[str, Any]]:
    """Aggregate canonical raw rows into minute-grain v1 feature rows."""

    if bucket_ms <= 0:
        raise ValueError("bucket_ms must be positive")
    ingest_ts = int(time.time() * 1000) if ingest_ts is None else ingest_ts
    quote_groups = _group_rows(_normalise_quote_rows(top_of_book_rows))
    depth_groups = _group_depth_snapshots(orderbook_rows)
    trade_groups = _group_rows(_normalise_trade_rows(trade_rows))
    keys = sorted(set(quote_groups) | set(depth_groups) | set(trade_groups))

    out: list[dict[str, Any]] = []
    for key in keys:
        quotes = _Series(quote_groups.get(key, []))
        depth = _Series(depth_groups.get(key, []))
        trades = _Series(trade_groups.get(key, []))
        feature_times = _feature_times(quotes.rows, depth.rows, trades.rows, bucket_ms=bucket_ms)
        for feature_ts in feature_times:
            if not _feature_ts_in_window(feature_ts, since_ms=since_ms, until_ms=until_ms):
                continue
            row = _build_feature_row(
                key=key,
                feature_ts=feature_ts,
                ingest_ts=ingest_ts,
                quotes=quotes,
                depth=depth,
                trades=trades,
                quality_config=quality_config,
            )
            out.append(row)
    out.sort(key=lambda row: (row["source"], row["source_symbol"], row["feature_ts"]))
    return out


def run_feature_batch(
    warehouse_dir: Path | str,
    *,
    max_rows_per_partition: int = 50_000,
    ingest_ts: int | None = None,
    quality_config: FeatureQualityConfig = DEFAULT_QUALITY_CONFIG,
    since_ms: int | None = None,
    until_ms: int | None = None,
    canonical_symbol_like: str | None = None,
    skip_existing: bool = False,
) -> FeatureBatchReport:
    """Read canonical warehouse tables and append ``features_15m_v1`` rows."""

    warehouse_dir = Path(warehouse_dir)
    raw_since_ms = None if since_ms is None else max(0, int(since_ms) - 30 * BUCKET_MS)
    raw_where_sql, raw_params = _raw_window_sql(
        since_ms=raw_since_ms,
        until_ms=until_ms,
        canonical_symbol_like=canonical_symbol_like,
    )
    with open_warehouse(warehouse_dir) as conn:
        top_of_book = _fetch_dicts(
            conn,
            "SELECT ts, source, source_symbol, source_market, canonical_symbol, "
            f"bid_price, ask_price, spread FROM raw_top_of_book{raw_where_sql}",
            raw_params,
        )
        orderbook = _fetch_dicts(
            conn,
            "SELECT ts, source, source_symbol, source_market, canonical_symbol, "
            f"side, level, price, size FROM raw_orderbook_snapshot{raw_where_sql}",
            raw_params,
        )
        trades = _fetch_dicts(
            conn,
            "SELECT ts, source, source_symbol, source_market, canonical_symbol, "
            f"price, size, side FROM raw_trades{raw_where_sql}",
            raw_params,
        )

    rows = aggregate_features_15m_v1(
        top_of_book_rows=top_of_book,
        orderbook_rows=orderbook,
        trade_rows=trades,
        ingest_ts=ingest_ts,
        quality_config=quality_config,
        since_ms=since_ms,
        until_ms=until_ms,
    )
    rows_generated = len(rows)
    rows = _filter_feature_window(rows, since_ms=since_ms, until_ms=until_ms)
    if skip_existing and rows:
        rows = _filter_new_feature_rows(warehouse_dir, rows)
    with WarehouseWriter(
        warehouse_dir,
        max_rows_per_partition=max_rows_per_partition,
    ) as writer:
        writer.append_rows("features_15m_v1", rows)
        writer.flush("features_15m_v1")
        rows_written = writer.stats.rows_written.get("features_15m_v1", 0)
    return FeatureBatchReport(rows_generated=rows_generated, rows_written=rows_written)


def _build_feature_row(
    *,
    key: tuple[str, str],
    feature_ts: int,
    ingest_ts: int,
    quotes: _Series,
    depth: _Series,
    trades: _Series,
    quality_config: FeatureQualityConfig,
) -> dict[str, Any]:
    source, source_symbol = key
    quote = quotes.latest_at(feature_ts)
    depth_snapshot = depth.latest_at(feature_ts)
    trade = trades.latest_at(feature_ts)
    bid = _as_float((quote or {}).get("bid_price"))
    ask = _as_float((quote or {}).get("ask_price"))
    if bid is None:
        bid = _depth_price(depth_snapshot, "BID", 0)
    if ask is None:
        ask = _depth_price(depth_snapshot, "ASK", 0)
    quality = compute_quality_fields(
        feature_ts=feature_ts,
        quote_ts=_quote_ts(quote, depth_snapshot),
        depth_ts=_row_ts(depth_snapshot),
        trade_ts=_row_ts(trade),
        config=quality_config,
    )
    context = _latest_context(feature_ts, quote, depth_snapshot, trade)
    spread = _spread(quote, bid, ask)
    mid = _mid(bid, ask)
    bid_l1 = _depth_size(depth_snapshot, "BID", 1)
    ask_l1 = _depth_size(depth_snapshot, "ASK", 1)
    bid_l5 = _depth_size(depth_snapshot, "BID", 5)
    ask_l5 = _depth_size(depth_snapshot, "ASK", 5)
    bid_l10 = _depth_size(depth_snapshot, "BID", 10)
    ask_l10 = _depth_size(depth_snapshot, "ASK", 10)
    window_trades_1m = trades.window(feature_ts - BUCKET_MS, feature_ts)
    signed_volume = _signed_volume(window_trades_1m)
    trade_volume = sum(_as_float(row.get("size")) or 0.0 for row in window_trades_1m)
    trade_count = len(window_trades_1m)
    ret_1m = _return_between(quotes, feature_ts, BUCKET_MS)
    ret_5m = _return_between(quotes, feature_ts, 5 * BUCKET_MS)
    ret_15m = _return_between(quotes, feature_ts, 15 * BUCKET_MS)
    ret_30m = _return_30m_or_fallback(
        quotes=quotes,
        feature_ts=feature_ts,
        ret_15m=ret_15m,
        ret_5m=ret_5m,
        ret_1m=ret_1m,
    )
    rv_1m = _realized_vol(quotes, feature_ts, BUCKET_MS)
    tick_obi_l1 = _imbalance_or_neutral(bid_l1, ask_l1)
    bid_l3 = _depth_size(depth_snapshot, "BID", 3)
    ask_l3 = _depth_size(depth_snapshot, "ASK", 3)
    tick_obi_l3 = _imbalance_or_neutral(bid_l3, ask_l3)
    market_features = _market_features(context.get("canonical_symbol"))

    return {
        "ts": feature_ts,
        "message_ts": feature_ts,
        "feature_ts": feature_ts,
        "ingest_ts": ingest_ts,
        "source": source,
        "source_symbol": source_symbol,
        "source_market": context.get("source_market"),
        "canonical_symbol": context.get("canonical_symbol"),
        "symbol": context.get("canonical_symbol") or source_symbol,
        "feature_version": FEATURE_VERSION,
        **quality,
        "spread": spread,
        "market_implied_prob": ask,
        "mid_price": mid,
        "microprice": _microprice(bid, ask, bid_l1, ask_l1),
        "obi_l1": tick_obi_l1,
        "obi_l5": _imbalance_or_neutral(bid_l5, ask_l5),
        "obi_l10": _imbalance_or_neutral(bid_l10, ask_l10),
        "signed_volume_1m": signed_volume,
        "trade_imbalance_1m": _ratio(signed_volume, trade_volume),
        "trade_count_1m": trade_count,
        "trade_volume_1m": trade_volume if window_trades_1m else 0.0,
        "ret_1m": ret_1m,
        "ret_5m": ret_5m,
        "ret_15m": ret_15m,
        "rv_1m": rv_1m,
        "rv_5m": _realized_vol(quotes, feature_ts, 5 * BUCKET_MS),
        "rv_15m": _realized_vol(quotes, feature_ts, 15 * BUCKET_MS),
        "minute_of_day": _minute_of_day(feature_ts),
        "day_of_week": _day_of_week(feature_ts),
        **market_features,
        "liquidity_bucket": _liquidity_bucket(spread, trade_volume),
        "ret_30m": ret_30m,
        "rv_30m": _realized_vol(quotes, feature_ts, 30 * BUCKET_MS),
        "aggressor_buy_ratio_1m": _aggressor_buy_ratio_or_neutral(window_trades_1m),
        "avg_trade_size_1m": _avg_trade_size_or_zero(trade_volume, trade_count),
        "tick_spread": spread,
        "tick_obi_l1": tick_obi_l1,
        "tick_obi_l3": tick_obi_l3,
        "tick_mid_price": mid,
        "tick_price_velocity": _tick_price_velocity(quotes, feature_ts, window_ms=5_000),
        "tick_trade_arrival_rate": _trade_arrival_rate(trades, feature_ts, window_ms=5_000),
    }


def _normalise_quote_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        ts = _as_int(row.get("ts"))
        source = row.get("source")
        source_symbol = row.get("source_symbol")
        if ts is None or not source or not source_symbol:
            continue
        out.append(
            {
                "ts": ts,
                "source": str(source),
                "source_symbol": str(source_symbol),
                "source_market": _optional_str(row.get("source_market")),
                "canonical_symbol": _optional_str(row.get("canonical_symbol")),
                "bid_price": _as_float(row.get("bid_price")),
                "ask_price": _as_float(row.get("ask_price")),
                "spread": _as_float(row.get("spread")),
            }
        )
    return out


def _normalise_trade_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        ts = _as_int(row.get("ts"))
        source = row.get("source")
        source_symbol = row.get("source_symbol")
        if ts is None or not source or not source_symbol:
            continue
        out.append(
            {
                "ts": ts,
                "source": str(source),
                "source_symbol": str(source_symbol),
                "source_market": _optional_str(row.get("source_market")),
                "canonical_symbol": _optional_str(row.get("canonical_symbol")),
                "price": _as_float(row.get("price")),
                "size": _as_float(row.get("size")),
                "side": _optional_str(row.get("side")),
            }
        )
    return out


def _group_rows(
    rows: Iterable[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["source"], row["source_symbol"])].append(row)
    return grouped


def _group_depth_snapshots(
    rows: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    snapshots: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        ts = _as_int(row.get("ts"))
        source = row.get("source")
        source_symbol = row.get("source_symbol")
        side = _optional_str(row.get("side"))
        level = _as_int(row.get("level"))
        price = _as_float(row.get("price"))
        size = _as_float(row.get("size"))
        if (
            ts is None
            or not source
            or not source_symbol
            or side not in ("BID", "ASK")
            or level is None
            or price is None
            or size is None
        ):
            continue
        snapshot_key = (str(source), str(source_symbol), ts)
        snapshot = snapshots.setdefault(
            snapshot_key,
            {
                "ts": ts,
                "source": str(source),
                "source_symbol": str(source_symbol),
                "source_market": _optional_str(row.get("source_market")),
                "canonical_symbol": _optional_str(row.get("canonical_symbol")),
                "BID": {},
                "ASK": {},
                "BID_PRICE": {},
                "ASK_PRICE": {},
            },
        )
        snapshot[side][level] = size
        snapshot[f"{side}_PRICE"][level] = price
    return _group_rows(snapshots.values())


def _feature_times(*groups: Iterable[dict[str, Any]], bucket_ms: int = BUCKET_MS) -> list[int]:
    times = set()
    for rows in groups:
        for row in rows:
            times.add(_ceil_bucket(int(row["ts"]), bucket_ms))
    return sorted(times)


def _ceil_minute(ts: int) -> int:
    return _ceil_bucket(ts, BUCKET_MS)


def _ceil_bucket(ts: int, bucket_ms: int) -> int:
    return ((ts + bucket_ms - 1) // bucket_ms) * bucket_ms


def _latest_context(
    feature_ts: int,
    *rows: dict[str, Any] | None,
) -> dict[str, str | None]:
    candidates = [
        row
        for row in rows
        if row is not None and _as_int(row.get("ts")) is not None and int(row["ts"]) <= feature_ts
    ]
    candidates.sort(key=lambda row: int(row["ts"]), reverse=True)
    for row in candidates:
        if row.get("canonical_symbol") or row.get("source_market"):
            return {
                "source_market": row.get("source_market"),
                "canonical_symbol": row.get("canonical_symbol"),
            }
    return {"source_market": None, "canonical_symbol": None}


def _spread(quote: dict[str, Any] | None, bid: float | None, ask: float | None) -> float | None:
    if quote is not None and quote.get("spread") is not None:
        return _as_float(quote.get("spread"))
    if bid is None or ask is None:
        return None
    return ask - bid


def _mid(bid: float | None, ask: float | None) -> float | None:
    if bid is None or ask is None:
        return None
    return (bid + ask) / 2.0


def _depth_size(snapshot: dict[str, Any] | None, side: str, levels: int) -> float | None:
    if snapshot is None:
        return None
    by_level = snapshot.get(side)
    if not isinstance(by_level, dict):
        return None
    total = sum(float(size) for level, size in by_level.items() if int(level) < levels)
    return total


def _depth_price(snapshot: dict[str, Any] | None, side: str, level: int) -> float | None:
    if snapshot is None:
        return None
    by_level = snapshot.get(f"{side}_PRICE")
    if not isinstance(by_level, dict):
        return None
    return _as_float(by_level.get(level))


def _quote_ts(quote: dict[str, Any] | None, depth_snapshot: dict[str, Any] | None) -> int | None:
    if _mid_from_quote(quote) is not None:
        return _row_ts(quote)
    if (
        _depth_price(depth_snapshot, "BID", 0) is not None
        and _depth_price(depth_snapshot, "ASK", 0) is not None
    ):
        return _row_ts(depth_snapshot)
    return None


def _microprice(
    bid: float | None,
    ask: float | None,
    bid_size: float | None,
    ask_size: float | None,
) -> float | None:
    if bid is None or ask is None or bid_size is None or ask_size is None:
        return None
    denom = bid_size + ask_size
    if denom <= 0:
        return None
    return (ask * bid_size + bid * ask_size) / denom


def _imbalance(bid_size: float | None, ask_size: float | None) -> float | None:
    if bid_size is None or ask_size is None:
        return None
    return _ratio(bid_size - ask_size, bid_size + ask_size)


def _imbalance_or_neutral(bid_size: float | None, ask_size: float | None) -> float:
    imbalance = _imbalance(bid_size, ask_size)
    return NEUTRAL_ORDERBOOK_IMBALANCE if imbalance is None else imbalance


def _signed_volume(rows: Iterable[Mapping[str, Any]]) -> float:
    total = 0.0
    for row in rows:
        size = _as_float(row.get("size")) or 0.0
        if row.get("side") == "BUY":
            total += size
        elif row.get("side") == "SELL":
            total -= size
    return total


def _aggressor_buy_ratio(rows: Sequence[Mapping[str, Any]]) -> float | None:
    if not rows:
        return None
    buy_count = sum(1 for row in rows if row.get("side") == "BUY")
    return buy_count / len(rows)


def _aggressor_buy_ratio_or_neutral(rows: Sequence[Mapping[str, Any]]) -> float:
    ratio = _aggressor_buy_ratio(rows)
    return NEUTRAL_AGGRESSOR_BUY_RATIO if ratio is None else ratio


def _avg_trade_size_or_zero(trade_volume: float, trade_count: int) -> float:
    avg = _ratio(trade_volume, trade_count)
    return NO_TRADE_AVG_SIZE if avg is None else avg


def _ratio(num: float | None, denom: float | None) -> float | None:
    if num is None or denom is None or denom == 0:
        return None
    return num / denom


def _return_between(quotes: _Series, feature_ts: int, window_ms: int) -> float | None:
    current = _mid_from_quote(quotes.latest_at(feature_ts))
    previous = _mid_from_quote(quotes.latest_at(feature_ts - window_ms))
    if current is None or previous is None or current <= 0 or previous <= 0:
        return None
    return math.log(current / previous)


def _return_30m_or_fallback(
    *,
    quotes: _Series,
    feature_ts: int,
    ret_15m: float | None,
    ret_5m: float | None,
    ret_1m: float | None,
) -> float:
    direct = _return_between(quotes, feature_ts, 30 * BUCKET_MS)
    if direct is not None:
        return direct
    if ret_15m is not None:
        return ret_15m * 2.0
    if ret_5m is not None:
        return ret_5m * 6.0
    if ret_1m is not None:
        return ret_1m * 30.0
    return NEUTRAL_RETURN


def _realized_vol(quotes: _Series, feature_ts: int, window_ms: int) -> float | None:
    start = feature_ts - window_ms
    points: list[tuple[int, float]] = []
    base = quotes.latest_at(start)
    base_mid = _mid_from_quote(base)
    if base is not None and base_mid is not None and base_mid > 0:
        points.append((int(base["ts"]), base_mid))
    for row in quotes.window(start, feature_ts):
        mid = _mid_from_quote(row)
        if mid is not None and mid > 0:
            points.append((int(row["ts"]), mid))
    deduped: list[tuple[int, float]] = []
    for ts, mid in points:
        if deduped and deduped[-1][0] == ts:
            deduped[-1] = (ts, mid)
        else:
            deduped.append((ts, mid))
    if len(deduped) < 2:
        return None
    sum_sq = 0.0
    for (_, prev), (_, curr) in zip(deduped, deduped[1:], strict=False):
        sum_sq += math.log(curr / prev) ** 2
    return math.sqrt(sum_sq)


def _tick_price_velocity(quotes: _Series, feature_ts: int, *, window_ms: int) -> float | None:
    current = _mid_from_quote(quotes.latest_at(feature_ts))
    previous = _mid_from_quote(quotes.latest_at(feature_ts - window_ms))
    if current is None or previous is None:
        return None
    return (current - previous) / (window_ms / 1000.0)


def _trade_arrival_rate(trades: _Series, feature_ts: int, *, window_ms: int) -> float:
    return len(trades.window(feature_ts - window_ms, feature_ts)) / (window_ms / 1000.0)


def _minute_of_day(feature_ts: int) -> float:
    dt = datetime.fromtimestamp(feature_ts / 1000, tz=UTC)
    return (dt.hour * 60 + dt.minute) / 1439.0


def _day_of_week(feature_ts: int) -> int:
    return datetime.fromtimestamp(feature_ts / 1000, tz=UTC).weekday()


def _market_features(canonical_symbol: str | None) -> dict[str, float | None]:
    if not canonical_symbol:
        return {"underlying_id": None, "horizon_minutes": None}
    family = canonical_symbol.split(":", 1)[0].upper()
    parts = family.split("-")
    underlying = parts[0] if parts else ""
    horizon_text = parts[1] if len(parts) > 1 else ""
    return {
        "underlying_id": _underlying_id(underlying),
        "horizon_minutes": _horizon_minutes(horizon_text),
    }


def _underlying_id(symbol: str) -> float | None:
    if not symbol:
        return None
    known = {"BTC": 1.0, "ETH": 2.0, "SOL": 3.0}
    if symbol in known:
        return known[symbol]
    return float(sum(ord(ch) for ch in symbol) % 997) + 10.0


def _horizon_minutes(text: str) -> float | None:
    upper = text.upper()
    if upper.endswith("M"):
        value = _as_float(upper[:-1])
        return value
    if upper.endswith("H"):
        value = _as_float(upper[:-1])
        return None if value is None else value * 60.0
    return _as_float(upper)


def _liquidity_bucket(spread: float | None, trade_volume: float) -> float | None:
    if spread is None:
        return None
    if spread <= 0.02 and trade_volume >= 50.0:
        return 3.0
    if spread <= 0.05 and trade_volume >= 10.0:
        return 2.0
    if spread <= 0.10:
        return 1.0
    return 0.0


def _mid_from_quote(row: dict[str, Any] | None) -> float | None:
    if row is None:
        return None
    return _mid(_as_float(row.get("bid_price")), _as_float(row.get("ask_price")))


def _row_ts(row: dict[str, Any] | None) -> int | None:
    if row is None:
        return None
    return _as_int(row.get("ts"))


def _raw_window_sql(
    *,
    since_ms: int | None,
    until_ms: int | None,
    canonical_symbol_like: str | None = None,
) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if since_ms is not None:
        clauses.append("ts >= ?")
        params.append(int(since_ms))
    if until_ms is not None:
        clauses.append("ts < ?")
        params.append(int(until_ms))
    if canonical_symbol_like:
        clauses.append("canonical_symbol LIKE ?")
        params.append(str(canonical_symbol_like))
    if not clauses:
        return "", []
    return " WHERE " + " AND ".join(clauses), params


def _filter_feature_window(
    rows: list[dict[str, Any]],
    *,
    since_ms: int | None,
    until_ms: int | None,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if (since_ms is None or int(row["feature_ts"]) >= int(since_ms))
        and (until_ms is None or int(row["feature_ts"]) < int(until_ms))
    ]


def _feature_ts_in_window(
    feature_ts: int,
    *,
    since_ms: int | None,
    until_ms: int | None,
) -> bool:
    return (since_ms is None or feature_ts >= int(since_ms)) and (
        until_ms is None or feature_ts < int(until_ms)
    )


def _filter_new_feature_rows(
    warehouse_dir: Path,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    min_ts = min(int(row["feature_ts"]) for row in rows)
    max_ts = max(int(row["feature_ts"]) for row in rows)
    with open_warehouse(warehouse_dir) as conn:
        try:
            existing = {
                (str(row[0]), str(row[1]), int(row[2]), str(row[3]))
                for row in conn.execute(
                    """
                    SELECT source, source_symbol, feature_ts, feature_version
                    FROM features_15m_v1
                    WHERE feature_ts >= ?
                      AND feature_ts <= ?
                    """,
                    [min_ts, max_ts],
                ).fetchall()
            }
        except (duckdb.CatalogException, duckdb.IOException):
            existing = set()
    return [
        row
        for row in rows
        if (
            str(row["source"]),
            str(row["source_symbol"]),
            int(row["feature_ts"]),
            str(row["feature_version"]),
        )
        not in existing
    ]


def _fetch_dicts(
    conn: duckdb.DuckDBPyConnection,
    query: str,
    params: Sequence[Any] = (),
) -> list[dict[str, Any]]:
    try:
        result = conn.execute(query, list(params))
    except (duckdb.CatalogException, duckdb.IOException):
        return []
    columns = [col[0] for col in result.description]
    return [dict(zip(columns, row, strict=True)) for row in result.fetchall()]


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None
