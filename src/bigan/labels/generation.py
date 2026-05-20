"""Independent 15-minute UP-token profitability labels (issue #9)."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib import error, parse, request

import duckdb
import orjson

from bigan.canonical.query import open_warehouse
from bigan.canonical.writer import WarehouseWriter

LABEL_SET_ID = "bigan-labels-15m-profitability"
LABEL_VERSION = "bigan-labels-15m-profitability-v1.1.0"
LABEL_KIND = "up_token_profitability"
HORIZON_MS = 15 * 60_000
DEFAULT_GAMMA_API_BASE = "https://gamma-api.polymarket.com"
DEFAULT_MARKET_SLUG_PREFIX = "btc-updown-15m-"
ROUND_LABEL_SOURCE = "polymarket_gamma_event_metadata_entry_ask_profitability"

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LabelBatchReport:
    """Summary of one ``labels_15m_v1`` batch."""

    rows_generated: int = 0
    rows_written: int = 0
    label_version: str = LABEL_VERSION


@dataclass(frozen=True, slots=True)
class _RoundPrice:
    round_start_ts: int
    round_end_ts: int
    ingest_ts: int
    start_price: float
    target_price: float
    input_index: int
    source_market: str | None = None
    round_slug: str | None = None
    label_source: str | None = ROUND_LABEL_SOURCE


def generate_labels_15m_v1(
    *,
    feature_rows: Iterable[Mapping[str, Any]],
    round_rows: Iterable[Mapping[str, Any]],
    ingest_ts: int | None = None,
    horizon_ms: int = HORIZON_MS,
    fee_bps: float = 0.0,
) -> list[dict[str, Any]]:
    """Generate Polymarket 15-minute UP-token trade labels.

    The label source of truth is Polymarket's round metadata, not an external
    spot-price stream. Gamma exposes the round's ``priceToBeat`` / ``finalPrice``
    pair for resolved rounds; those become ``start_price`` and ``target_price``.
    The model target is whether buying the UP token at the feature row's
    ``market_implied_prob`` would be profitable at settlement after entry fees.
    Rows with ``ingest_ts`` after this label run are ignored so a historical run
    cannot accidentally consume a later-corrected round row.
    """

    if horizon_ms <= 0:
        raise ValueError("horizon_ms must be positive")
    if fee_bps < 0.0:
        raise ValueError("fee_bps must be non-negative")

    run_ingest_ts = int(time.time() * 1000) if ingest_ts is None else int(ingest_ts)
    rounds = _normalise_rounds(round_rows)
    out: list[dict[str, Any]] = []

    for feature in _normalise_features(feature_rows):
        if not _is_up_token_feature(feature):
            continue
        feature_ts = int(feature["feature_ts"])
        entry_ask_price = _as_float(feature.get("market_implied_prob"))
        if entry_ask_price is None or entry_ask_price < 0.0 or entry_ask_price > 1.0:
            continue
        round_price = _select_round(
            rounds,
            feature_ts=feature_ts,
            source_market=feature.get("source_market"),
            run_ingest_ts=run_ingest_ts,
        )
        if round_price is None:
            continue

        direction_up_15m = _direction_up(round_price)
        settlement_price = 1.0 if direction_up_15m else 0.0
        entry_fee = entry_ask_price * (fee_bps / 10_000.0)
        entry_cost = entry_ask_price + entry_fee
        realized_return = settlement_price - entry_cost
        label_profit_up_15m = realized_return > 0.0
        out.append(
            {
                "ts": feature_ts,
                "message_ts": feature_ts,
                "feature_ts": feature_ts,
                "target_ts": round_price.round_end_ts,
                "ingest_ts": run_ingest_ts,
                "source": feature["source"],
                "source_symbol": feature["source_symbol"],
                "source_market": feature.get("source_market") or round_price.source_market,
                "canonical_symbol": feature.get("canonical_symbol"),
                "symbol": feature["symbol"],
                "label_version": LABEL_VERSION,
                "label_kind": LABEL_KIND,
                "round_slug": round_price.round_slug,
                "round_start_ts": round_price.round_start_ts,
                "round_end_ts": round_price.round_end_ts,
                "start_price": round_price.start_price,
                "target_price": round_price.target_price,
                "direction_up_15m": direction_up_15m,
                "entry_ask_price": entry_ask_price,
                "settlement_price": settlement_price,
                "entry_fee": entry_fee,
                "entry_cost": entry_cost,
                "realized_return": realized_return,
                "fee_bps": fee_bps,
                "label_profit_up_15m": label_profit_up_15m,
                "label_up_15m": label_profit_up_15m,
                "label_source": round_price.label_source,
            }
        )

    out.sort(key=lambda row: (row["source"], row["source_symbol"], row["feature_ts"]))
    return out


def run_label_batch(
    warehouse_dir: Path | str,
    *,
    max_rows_per_partition: int = 50_000,
    ingest_ts: int | None = None,
    horizon_ms: int = HORIZON_MS,
    fee_bps: float = 0.0,
    round_rows: Iterable[Mapping[str, Any]] | None = None,
    gamma_api_base: str = DEFAULT_GAMMA_API_BASE,
    market_slug_prefix: str = DEFAULT_MARKET_SLUG_PREFIX,
    request_timeout_seconds: float = 10.0,
) -> LabelBatchReport:
    """Read feature rows and append independent ``labels_15m_v1`` rows."""

    warehouse_dir = Path(warehouse_dir)
    run_ingest_ts = int(time.time() * 1000) if ingest_ts is None else int(ingest_ts)
    with open_warehouse(warehouse_dir) as conn:
        features = _fetch_feature_rows_for_labels(conn)

    resolved_round_rows = (
        list(round_rows)
        if round_rows is not None
        else fetch_polymarket_round_rows_for_features(
            features,
            ingest_ts=run_ingest_ts,
            gamma_api_base=gamma_api_base,
            market_slug_prefix=market_slug_prefix,
            horizon_ms=horizon_ms,
            request_timeout_seconds=request_timeout_seconds,
        )
    )

    rows = generate_labels_15m_v1(
        feature_rows=features,
        round_rows=resolved_round_rows,
        ingest_ts=run_ingest_ts,
        horizon_ms=horizon_ms,
        fee_bps=fee_bps,
    )
    with WarehouseWriter(
        warehouse_dir,
        max_rows_per_partition=max_rows_per_partition,
    ) as writer:
        writer.append_rows("labels_15m_v1", rows)
        writer.flush("labels_15m_v1")
        rows_written = writer.stats.rows_written.get("labels_15m_v1", 0)
    return LabelBatchReport(rows_generated=len(rows), rows_written=rows_written)


def fetch_polymarket_round_rows_for_features(
    feature_rows: Iterable[Mapping[str, Any]],
    *,
    ingest_ts: int | None = None,
    gamma_api_base: str = DEFAULT_GAMMA_API_BASE,
    market_slug_prefix: str = DEFAULT_MARKET_SLUG_PREFIX,
    horizon_ms: int = HORIZON_MS,
    request_timeout_seconds: float = 10.0,
) -> list[dict[str, Any]]:
    """Fetch resolved Polymarket round metadata needed by ``labels_15m_v1``."""

    if horizon_ms <= 0:
        raise ValueError("horizon_ms must be positive")
    run_ingest_ts = int(time.time() * 1000) if ingest_ts is None else int(ingest_ts)
    slugs = sorted(
        {
            slug
            for feature in _normalise_features(feature_rows)
            for slug in _candidate_round_slugs_for_feature(
                feature,
                market_slug_prefix=market_slug_prefix,
                horizon_ms=horizon_ms,
            )
        }
    )

    rows: list[dict[str, Any]] = []
    for slug in slugs:
        event = _fetch_gamma_event_by_slug(
            gamma_api_base,
            slug,
            request_timeout_seconds=request_timeout_seconds,
        )
        if event is None:
            continue
        row = polymarket_round_row_from_event(event, ingest_ts=run_ingest_ts)
        if row is not None:
            rows.append(row)
    return rows


def polymarket_round_row_from_event(
    event: Mapping[str, Any],
    *,
    ingest_ts: int,
) -> dict[str, Any] | None:
    """Map one Gamma event payload to the round-price row consumed by labels."""

    metadata = event.get("eventMetadata")
    if not isinstance(metadata, Mapping):
        return None
    start_price = _first_float(
        metadata,
        "priceToBeat",
        "price_to_beat",
        "openPrice",
        "open_price",
        "start_price",
    )
    target_price = _first_float(
        metadata,
        "finalPrice",
        "final_price",
        "closePrice",
        "close_price",
        "target_price",
    )
    if start_price is None or target_price is None:
        return None

    market = _first_market(event)
    round_start_ts = _first_ts_ms(event, "startTime", "eventStartTime", "round_start_ts")
    if round_start_ts is None and market is not None:
        round_start_ts = _first_ts_ms(
            market,
            "eventStartTime",
            "startTime",
            "round_start_ts",
        )
    round_end_ts = _first_ts_ms(event, "endDate", "round_end_ts", "target_ts")
    if round_end_ts is None and market is not None:
        round_end_ts = _first_ts_ms(market, "endDate", "round_end_ts", "target_ts")
    if round_start_ts is None or round_end_ts is None:
        return None

    source_market = None
    if market is not None:
        source_market = _optional_str(
            market.get("conditionId")
            or market.get("condition_id")
            or market.get("source_market")
        )
    slug = _optional_str(event.get("slug") or event.get("ticker"))

    return {
        "ts": round_start_ts,
        "message_ts": round_start_ts,
        "ingest_ts": int(ingest_ts),
        "source": "polymarket",
        "source_market": source_market,
        "round_slug": slug,
        "round_start_ts": round_start_ts,
        "round_end_ts": round_end_ts,
        "start_price": start_price,
        "target_price": target_price,
        "label_source": ROUND_LABEL_SOURCE,
    }


def _normalise_features(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        feature_ts = _as_int(row.get("feature_ts") or row.get("ts"))
        source = _optional_str(row.get("source"))
        source_symbol = _optional_str(row.get("source_symbol"))
        if feature_ts is None or source is None or source_symbol is None:
            continue
        canonical_symbol = _optional_str(row.get("canonical_symbol"))
        symbol = _optional_str(row.get("symbol")) or canonical_symbol or source_symbol
        out.append(
            {
                "feature_ts": feature_ts,
                "source": source,
                "source_symbol": source_symbol,
                "source_market": _optional_str(row.get("source_market")),
                "canonical_symbol": canonical_symbol,
                "symbol": symbol,
                "market_implied_prob": _as_float(row.get("market_implied_prob")),
            }
        )
    return out


def _normalise_rounds(rows: Iterable[Mapping[str, Any]]) -> list[_RoundPrice]:
    out: list[_RoundPrice] = []
    for idx, row in enumerate(rows):
        round_start_ts = _first_ts_ms(
            row,
            "round_start_ts",
            "start_ts",
            "startTime",
            "eventStartTime",
        )
        round_end_ts = _first_ts_ms(row, "round_end_ts", "target_ts", "end_ts", "endDate")
        start_price = _first_float(
            row,
            "start_price",
            "price_to_beat",
            "priceToBeat",
            "open_price",
            "openPrice",
        )
        target_price = _first_float(
            row,
            "target_price",
            "final_price",
            "finalPrice",
            "close_price",
            "closePrice",
        )
        if (
            round_start_ts is None
            or round_end_ts is None
            or start_price is None
            or target_price is None
        ):
            continue
        out.append(
            _RoundPrice(
                round_start_ts=round_start_ts,
                round_end_ts=round_end_ts,
                ingest_ts=_as_int(row.get("ingest_ts")) or 0,
                start_price=start_price,
                target_price=target_price,
                input_index=idx,
                source_market=_optional_str(
                    row.get("source_market")
                    or row.get("conditionId")
                    or row.get("condition_id")
                ),
                round_slug=_optional_str(row.get("round_slug") or row.get("slug")),
                label_source=_optional_str(row.get("label_source")) or ROUND_LABEL_SOURCE,
            )
        )
    out.sort(
        key=lambda round_: (
            round_.round_start_ts,
            round_.round_end_ts,
            round_.source_market or "",
            round_.ingest_ts,
            round_.input_index,
        )
    )
    return out


def _select_round(
    rounds: Iterable[_RoundPrice],
    *,
    feature_ts: int,
    source_market: str | None,
    run_ingest_ts: int,
) -> _RoundPrice | None:
    market = _optional_str(source_market)
    eligible = [
        round_
        for round_ in rounds
        if round_.ingest_ts <= run_ingest_ts
    ]
    if market is not None:
        market_matches = [
            round_
            for round_ in eligible
            if round_.source_market == market and feature_ts < round_.round_end_ts
        ]
        if market_matches:
            return max(
                market_matches,
                key=lambda round_: (
                    round_.ingest_ts,
                    round_.round_start_ts,
                    round_.input_index,
                ),
            )
        return None
    eligible = [
        round_
        for round_ in eligible
        if round_.round_start_ts <= feature_ts < round_.round_end_ts
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda round_: (
            round_.ingest_ts,
            round_.round_start_ts,
            round_.input_index,
        ),
    )


def _fetch_feature_rows_for_labels(conn: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    return _fetch_dicts(
        conn,
        "SELECT feature_ts, source, source_symbol, source_market, "
        "canonical_symbol, symbol, market_implied_prob FROM features_15m_v1",
    )


def _is_up_token_feature(feature: Mapping[str, Any]) -> bool:
    canonical_symbol = _optional_str(feature.get("canonical_symbol"))
    if canonical_symbol is None:
        return False
    text = canonical_symbol.upper()
    return text.endswith(":UP") or text.endswith("-UP-15M")


def _direction_up(round_price: _RoundPrice) -> bool:
    return round_price.target_price >= round_price.start_price


def _candidate_round_slugs(
    feature_ts: int,
    *,
    market_slug_prefix: str,
    horizon_ms: int,
) -> tuple[str, ...]:
    current_start = (feature_ts // horizon_ms) * horizon_ms
    previous_start = ((max(0, feature_ts - 1)) // horizon_ms) * horizon_ms
    starts = sorted({current_start, previous_start})
    return tuple(f"{market_slug_prefix}{start // 1000}" for start in starts)


def _candidate_round_slugs_for_feature(
    feature: Mapping[str, Any],
    *,
    market_slug_prefix: str,
    horizon_ms: int,
) -> tuple[str, ...]:
    slugs = list(
        _candidate_round_slugs(
            int(feature["feature_ts"]),
            market_slug_prefix=market_slug_prefix,
            horizon_ms=horizon_ms,
        )
    )
    canonical_slug = _canonical_round_slug(
        feature.get("canonical_symbol") or feature.get("symbol"),
        market_slug_prefix=market_slug_prefix,
    )
    if canonical_slug is not None and canonical_slug not in slugs:
        slugs.append(canonical_slug)
    return tuple(slugs)


def _canonical_round_slug(
    value: Any,
    *,
    market_slug_prefix: str,
) -> str | None:
    text = _optional_str(value)
    if text is None:
        return None
    for part in text.split(":"):
        if part.startswith(market_slug_prefix):
            return part
    return None


def _fetch_gamma_event_by_slug(
    gamma_api_base: str,
    slug: str,
    *,
    request_timeout_seconds: float,
) -> Mapping[str, Any] | None:
    base = gamma_api_base.rstrip("/")
    url = f"{base}/events/slug/{parse.quote(slug, safe='')}"
    req = request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "BiGan-labels/1.0",
        },
    )
    try:
        with request.urlopen(req, timeout=request_timeout_seconds) as resp:
            payload = orjson.loads(resp.read())
    except (TimeoutError, error.URLError, orjson.JSONDecodeError) as exc:
        logger.warning(
            "labels.gamma_event_fetch_failed",
            extra={"slug": slug, "err": str(exc)},
        )
        return None
    return payload if isinstance(payload, Mapping) else None


def _first_market(event: Mapping[str, Any]) -> Mapping[str, Any] | None:
    markets = event.get("markets")
    if isinstance(markets, list) and markets and isinstance(markets[0], Mapping):
        return markets[0]
    return None


def _fetch_dicts(conn: duckdb.DuckDBPyConnection, sql: str) -> list[dict[str, Any]]:
    try:
        result = conn.execute(sql)
    except (duckdb.BinderException, duckdb.CatalogException, duckdb.IOException):
        return []
    columns = [col[0] for col in result.description]
    return [dict(zip(columns, row, strict=True)) for row in result.fetchall()]


def _first_float(row: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _as_float(row.get(key))
        if value is not None:
            return value
    return None


def _first_ts_ms(row: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = _ts_ms(row.get(key))
        if value is not None:
            return value
    return None


def _ts_ms(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
