"""Independent 15-minute UP-token profitability labels (issue #9)."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib import error, parse, request

import duckdb
import orjson

from bigan.canonical.query import open_warehouse
from bigan.canonical.writer import WarehouseWriter
from bigan.labels.v6 import empty_volatility_fields, settlement_3way_label, settlement_margin

LABEL_SET_ID = "bigan-labels-15m-profitability"
LABEL_VERSION = "bigan-labels-15m-profitability-v1.2.0"
UP_LABEL_KIND = "up_token_profitability"
DOWN_LABEL_KIND = "down_token_profitability"
LABEL_KIND = UP_LABEL_KIND
HORIZON_MS = 15 * 60_000
DEFAULT_GAMMA_API_BASE = "https://gamma-api.polymarket.com"
DEFAULT_MARKET_SLUG_PREFIX = "btc-updown-15m-"
ROUND_LABEL_SOURCE = "polymarket_gamma_event_metadata_entry_ask_profitability"
MARKET_OUTCOME_PRICE_LABEL_SOURCE = "polymarket_gamma_market_outcome_prices_entry_ask_profitability"

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LabelBatchReport:
    """Summary of one ``labels_15m_v1`` batch."""

    rows_generated: int = 0
    rows_written: int = 0
    label_version: str = LABEL_VERSION
    monitoring_outcomes_written: int = 0


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
    settlement_neutral_margin: float = 0.0,
) -> list[dict[str, Any]]:
    """Generate Polymarket 15-minute outcome-token trade labels.

    The label source of truth is Polymarket's round metadata, not an external
    spot-price stream. Gamma exposes the round's ``priceToBeat`` / ``finalPrice``
    pair for resolved rounds; those become ``start_price`` and ``target_price``.
    The model target is whether buying the UP token at the feature row's
    ``market_implied_prob`` would be profitable at settlement after entry fees.
    UP rows populate ``label_profit_up_15m``; DOWN rows populate
    ``label_profit_down_15m``.
    V6 settlement rows also include ``label_settlement_3way``. Because this
    generator only consumes round metadata, it marks volatility path fields as
    ``missing_price_path``; raw top-of-book coverage must be supplied by a
    separate corpus builder before ``label_volatility_*`` can be non-null.
    Rows with ``ingest_ts`` after this label run are ignored so a historical run
    cannot accidentally consume a later-corrected round row.
    """

    if horizon_ms <= 0:
        raise ValueError("horizon_ms must be positive")
    if fee_bps < 0.0:
        raise ValueError("fee_bps must be non-negative")
    if settlement_neutral_margin < 0.0:
        raise ValueError("settlement_neutral_margin must be non-negative")

    run_ingest_ts = int(time.time() * 1000) if ingest_ts is None else int(ingest_ts)
    rounds = _normalise_rounds(round_rows)
    out: list[dict[str, Any]] = []

    for feature in _normalise_features(feature_rows):
        outcome_side = _outcome_side_for_feature(feature)
        if outcome_side not in {"UP", "DOWN"}:
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
        settlement_price = _settlement_price_for_side(direction_up_15m, outcome_side)
        entry_fee = entry_ask_price * (fee_bps / 10_000.0)
        entry_cost = entry_ask_price + entry_fee
        realized_return = settlement_price - entry_cost
        margin = settlement_margin(round_price.start_price, round_price.target_price)
        label_profit = realized_return > 0.0
        label_profit_up_15m = label_profit if outcome_side == "UP" else None
        label_profit_down_15m = label_profit if outcome_side == "DOWN" else None
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
                "label_kind": UP_LABEL_KIND if outcome_side == "UP" else DOWN_LABEL_KIND,
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
                "settlement_margin": margin,
                "settlement_abs_margin": abs(margin),
                "settlement_neutral_margin": settlement_neutral_margin,
                "label_settlement_3way": settlement_3way_label(
                    round_price.start_price,
                    round_price.target_price,
                    neutral_margin_abs=settlement_neutral_margin,
                ),
                **empty_volatility_fields(),
                "label_profit_up_15m": label_profit_up_15m,
                "label_profit_down_15m": label_profit_down_15m,
                "label_up_15m": bool(label_profit_up_15m),
                "label_down_15m": label_profit_down_15m,
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
    settlement_neutral_margin: float = 0.0,
    round_rows: Iterable[Mapping[str, Any]] | None = None,
    gamma_api_base: str = DEFAULT_GAMMA_API_BASE,
    market_slug_prefix: str = DEFAULT_MARKET_SLUG_PREFIX,
    request_timeout_seconds: float = 10.0,
    request_concurrency: int = 8,
    monitoring_db_path: Path | str | None = None,
    monitoring_model_version: str | None = None,
    skip_existing_labels: bool = False,
    since_ms: int | None = None,
    until_ms: int | None = None,
) -> LabelBatchReport:
    """Read feature rows and append independent ``labels_15m_v1`` rows."""

    warehouse_dir = Path(warehouse_dir)
    run_ingest_ts = int(time.time() * 1000) if ingest_ts is None else int(ingest_ts)
    with open_warehouse(warehouse_dir) as conn:
        features = _fetch_feature_rows_for_labels(conn, since_ms=since_ms, until_ms=until_ms)

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
            request_concurrency=request_concurrency,
        )
    )

    rows = generate_labels_15m_v1(
        feature_rows=features,
        round_rows=resolved_round_rows,
        ingest_ts=run_ingest_ts,
        horizon_ms=horizon_ms,
        fee_bps=fee_bps,
        settlement_neutral_margin=settlement_neutral_margin,
    )
    rows_generated = len(rows)
    if skip_existing_labels and rows:
        rows = _filter_new_label_rows(warehouse_dir, rows)
    with WarehouseWriter(
        warehouse_dir,
        max_rows_per_partition=max_rows_per_partition,
    ) as writer:
        writer.append_rows("labels_15m_v1", rows)
        writer.flush("labels_15m_v1")
        rows_written = writer.stats.rows_written.get("labels_15m_v1", 0)
    monitoring_outcomes_written = 0
    if monitoring_db_path is not None and monitoring_model_version is not None and rows:
        from bigan.mlops.registry import connect_mlops_db, initialize_mlops_db
        from bigan.monitoring import record_label_rows_as_outcomes

        conn = connect_mlops_db(monitoring_db_path)
        try:
            initialize_mlops_db(conn)
            monitoring_outcomes_written = record_label_rows_as_outcomes(
                conn,
                rows,
                model_version=monitoring_model_version,
            )
        finally:
            conn.close()
    return LabelBatchReport(
        rows_generated=rows_generated,
        rows_written=rows_written,
        monitoring_outcomes_written=monitoring_outcomes_written,
    )


def fetch_polymarket_round_rows_for_features(
    feature_rows: Iterable[Mapping[str, Any]],
    *,
    ingest_ts: int | None = None,
    gamma_api_base: str = DEFAULT_GAMMA_API_BASE,
    market_slug_prefix: str = DEFAULT_MARKET_SLUG_PREFIX,
    horizon_ms: int = HORIZON_MS,
    request_timeout_seconds: float = 10.0,
    request_concurrency: int = 8,
) -> list[dict[str, Any]]:
    """Fetch resolved Polymarket round metadata needed by ``labels_15m_v1``."""

    if horizon_ms <= 0:
        raise ValueError("horizon_ms must be positive")
    if request_concurrency <= 0:
        raise ValueError("request_concurrency must be positive")
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

    def fetch_slug(slug: str) -> dict[str, Any] | None:
        event = _fetch_gamma_event_by_slug(
            gamma_api_base,
            slug,
            request_timeout_seconds=request_timeout_seconds,
        )
        row = (
            None
            if event is None
            else polymarket_round_row_from_event(event, ingest_ts=run_ingest_ts)
        )
        if row is None:
            market = _fetch_gamma_market_by_slug(
                gamma_api_base,
                slug,
                request_timeout_seconds=request_timeout_seconds,
            )
            row = (
                None
                if market is None
                else polymarket_round_row_from_market(market, ingest_ts=run_ingest_ts)
            )
        return row

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(request_concurrency, max(1, len(slugs)))) as executor:
        for row in executor.map(fetch_slug, slugs):
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


def polymarket_round_row_from_market(
    market: Mapping[str, Any],
    *,
    ingest_ts: int,
) -> dict[str, Any] | None:
    """Map one Gamma market payload to a resolved round-price row.

    Resolved short-horizon markets expose the winning outcome through
    ``outcomePrices`` even when the event metadata endpoint is unavailable.
    The BTC start/final oracle prices are not present in this payload, so the
    row uses a synthetic 0.5 -> {0, 1} price pair solely to preserve the
    existing binary direction contract.
    """

    if not _is_resolved_market(market):
        return None
    slug = _optional_str(market.get("slug"))
    source_market = _optional_str(
        market.get("conditionId")
        or market.get("condition_id")
        or market.get("source_market")
    )
    if slug is None or source_market is None:
        return None
    round_start_ts = _round_start_from_slug(slug)
    if round_start_ts is None:
        round_start_ts = _first_ts_ms(market, "eventStartTime", "startTime", "round_start_ts")
    round_end_ts = _first_ts_ms(market, "endDate", "round_end_ts", "target_ts")
    if round_start_ts is None or round_end_ts is None:
        return None

    up_won = _up_outcome_won(market)
    if up_won is None:
        return None

    return {
        "ts": round_start_ts,
        "message_ts": round_start_ts,
        "ingest_ts": int(ingest_ts),
        "source": "polymarket",
        "source_market": source_market,
        "round_slug": slug,
        "round_start_ts": round_start_ts,
        "round_end_ts": round_end_ts,
        "start_price": 0.5,
        "target_price": 1.0 if up_won else 0.0,
        "label_source": MARKET_OUTCOME_PRICE_LABEL_SOURCE,
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


def _fetch_feature_rows_for_labels(
    conn: duckdb.DuckDBPyConnection,
    *,
    since_ms: int | None = None,
    until_ms: int | None = None,
) -> list[dict[str, Any]]:
    clauses = ["TRUE"]
    params: list[int] = []
    if since_ms is not None:
        clauses.append("feature_ts >= ?")
        params.append(int(since_ms))
    if until_ms is not None:
        clauses.append("feature_ts < ?")
        params.append(int(until_ms))
    where_sql = " AND ".join(clauses)
    return _fetch_dicts(
        conn,
        "SELECT feature_ts, source, source_symbol, source_market, "
        "canonical_symbol, symbol, market_implied_prob FROM features_15m_v1 "
        f"WHERE {where_sql}",
        params,
    )


def _filter_new_label_rows(
    warehouse_dir: Path | str,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not rows:
        return []
    with open_warehouse(warehouse_dir) as conn:
        existing = _fetch_existing_label_keys(conn)
    return [
        row
        for row in rows
        if (
            int(row["feature_ts"]),
            str(row["source"]),
            str(row["source_symbol"]),
            str(row["label_version"]),
            str(row["label_kind"]),
        )
        not in existing
    ]


def _fetch_existing_label_keys(
    conn: duckdb.DuckDBPyConnection,
) -> set[tuple[int, str, str, str, str]]:
    try:
        result = conn.execute(
            """
            SELECT feature_ts, source, source_symbol, label_version, label_kind
            FROM labels_15m_v1
            WHERE label_version = ?
            """,
            [LABEL_VERSION],
        )
    except (duckdb.BinderException, duckdb.CatalogException, duckdb.IOException):
        return set()
    return {
        (int(feature_ts), str(source), str(source_symbol), str(label_version), str(label_kind))
        for feature_ts, source, source_symbol, label_version, label_kind in result.fetchall()
    }


def _outcome_side_for_feature(feature: Mapping[str, Any]) -> str | None:
    canonical_symbol = _optional_str(feature.get("canonical_symbol"))
    if canonical_symbol is None:
        return None
    text = canonical_symbol.upper()
    if text.endswith(":UP") or text.endswith("-UP-15M"):
        return "UP"
    if text.endswith(":DOWN") or text.endswith("-DOWN-15M"):
        return "DOWN"
    return None


def _direction_up(round_price: _RoundPrice) -> bool:
    return round_price.target_price >= round_price.start_price


def _settlement_price_for_side(direction_up: bool, outcome_side: str) -> float:
    if outcome_side == "UP":
        return 1.0 if direction_up else 0.0
    if outcome_side == "DOWN":
        return 0.0 if direction_up else 1.0
    raise ValueError("outcome_side must be UP or DOWN")


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
    resolved_prefix = _market_slug_prefix_for_feature(feature, default=market_slug_prefix)
    resolved_horizon_ms = _horizon_ms_for_feature(feature, default=horizon_ms)
    slugs = list(
        _candidate_round_slugs(
            int(feature["feature_ts"]),
            market_slug_prefix=resolved_prefix,
            horizon_ms=resolved_horizon_ms,
        )
    )
    canonical_slug = _canonical_round_slug(
        feature.get("canonical_symbol") or feature.get("symbol"),
        market_slug_prefix=resolved_prefix,
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
        if "-updown-" in part:
            return part
    return None


def _market_slug_prefix_for_feature(
    feature: Mapping[str, Any],
    *,
    default: str,
) -> str:
    slug = _canonical_round_slug(
        feature.get("canonical_symbol") or feature.get("symbol"),
        market_slug_prefix=default,
    )
    if slug is None or "-" not in slug:
        return default
    return slug.rsplit("-", 1)[0] + "-"


def _horizon_ms_for_feature(
    feature: Mapping[str, Any],
    *,
    default: int,
) -> int:
    text = _optional_str(feature.get("canonical_symbol") or feature.get("symbol"))
    if text is None:
        return default
    family = text.split(":", 1)[0].upper()
    parts = family.split("-")
    if len(parts) < 2:
        return default
    parsed = _horizon_text_to_ms(parts[1])
    return parsed or default


def _horizon_text_to_ms(text: str) -> int | None:
    upper = text.strip().upper()
    if len(upper) < 2:
        return None
    value_text = upper[:-1]
    suffix = upper[-1]
    if not value_text.isdigit():
        return None
    value = int(value_text)
    if suffix == "M":
        return value * 60_000
    if suffix == "H":
        return value * 60 * 60_000
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
            "labels.gamma_event_fetch_failed slug=%s err=%s",
            slug,
            exc,
            extra={"slug": slug, "err": str(exc)},
        )
        return None
    return payload if isinstance(payload, Mapping) else None


def _fetch_gamma_market_by_slug(
    gamma_api_base: str,
    slug: str,
    *,
    request_timeout_seconds: float,
) -> Mapping[str, Any] | None:
    base = gamma_api_base.rstrip("/")
    param_sets = (
        {"slug": slug, "closed": "true", "limit": "1"},
        {"slug": slug, "active": "true", "closed": "false", "limit": "1"},
        {"slug": slug, "limit": "1"},
    )
    errors: list[str] = []
    for params in param_sets:
        url = f"{base}/markets?{parse.urlencode(params)}"
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
            errors.append(str(exc))
            continue
        if isinstance(payload, list) and payload and isinstance(payload[0], Mapping):
            return payload[0]
    if errors:
        logger.warning(
            "labels.gamma_market_fetch_failed slug=%s failed_attempts=%d last_err=%s",
            slug,
            len(errors),
            errors[-1],
            extra={"slug": slug, "failed_attempts": len(errors), "err": errors[-1]},
        )
    return None


def _is_resolved_market(market: Mapping[str, Any]) -> bool:
    if bool(market.get("closed")):
        return True
    status = _optional_str(market.get("umaResolutionStatus"))
    return status is not None and status.lower() == "resolved"


def _up_outcome_won(market: Mapping[str, Any]) -> bool | None:
    outcomes = _json_list(market.get("outcomes"))
    prices = _json_list(market.get("outcomePrices"))
    if len(outcomes) != len(prices) or not outcomes:
        return None
    up_price = None
    down_price = None
    for outcome, price in zip(outcomes, prices, strict=True):
        normalized = str(outcome).strip().upper()
        parsed_price = _as_float(price)
        if parsed_price is None:
            return None
        if normalized == "UP":
            up_price = parsed_price
        elif normalized == "DOWN":
            down_price = parsed_price
    if up_price is None or down_price is None:
        return None
    if max(up_price, down_price) < 0.99:
        return None
    return up_price > down_price


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = orjson.loads(value)
        except orjson.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _round_start_from_slug(slug: str) -> int | None:
    try:
        start_seconds = int(slug.rsplit("-", 1)[-1])
    except ValueError:
        return None
    return start_seconds * 1000


def _first_market(event: Mapping[str, Any]) -> Mapping[str, Any] | None:
    markets = event.get("markets")
    if isinstance(markets, list) and markets and isinstance(markets[0], Mapping):
        return markets[0]
    return None


def _fetch_dicts(
    conn: duckdb.DuckDBPyConnection,
    sql: str,
    params: list[int] | None = None,
) -> list[dict[str, Any]]:
    try:
        result = conn.execute(sql, params or [])
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
