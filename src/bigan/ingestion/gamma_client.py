"""Async client for the Polymarket Gamma API.

Used to discover the active set of BTC 15-minute up/down markets so the WS
client can dynamically (un)subscribe as markets open and resolve every 15 min.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import aiohttp
import orjson

from bigan.canonical.symbols import symbol_mapping_row

from .metrics import GAMMA_POLLS_TOTAL

logger = logging.getLogger(__name__)


_GAMMA_MAX_PAGE_LIMIT = 100
BTC_15M_HORIZON_MS = 15 * 60_000
POLYMARKET_SOURCE = "polymarket"
BTC_15M_SYMBOL_KIND = "btc_15m_outcome"


@dataclass(frozen=True, slots=True)
class MarketDiscoverySpec:
    """Configuration for one Polymarket up/down market family."""

    slug_prefix: str
    underlying: str
    horizon_ms: int
    symbol_kind: str


DEFAULT_MARKET_DISCOVERY_SPEC = MarketDiscoverySpec(
    slug_prefix="btc-updown-15m-",
    underlying="BTC",
    horizon_ms=BTC_15M_HORIZON_MS,
    symbol_kind=BTC_15M_SYMBOL_KIND,
)


@dataclass(frozen=True, slots=True)
class ActiveMarket:
    """A minimal record of one active CLOB market relevant to ingestion."""

    slug: str
    condition_id: str
    asset_id_up: str
    asset_id_down: str
    start_ts_ms: int
    end_ts_ms: int
    tick_size: str
    underlying: str = "BTC"
    horizon_ms: int = BTC_15M_HORIZON_MS
    symbol_kind: str = BTC_15M_SYMBOL_KIND

    @property
    def asset_ids(self) -> tuple[str, str]:
        return (self.asset_id_up, self.asset_id_down)


def _parse_iso8601_to_ms(s: str | None) -> int:
    """Parse an ISO-8601 timestamp (with optional ``Z``) into ms epoch.

    Returns 0 if input is falsy. Gamma uses UTC ``Z``-suffixed strings.
    """
    if not s:
        return 0
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        logger.warning("gamma.parse_iso8601_failed", extra={"raw": s})
        return 0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def _market_from_gamma(
    record: dict[str, Any],
    spec: MarketDiscoverySpec | None = None,
) -> ActiveMarket | None:
    """Convert one Gamma ``/markets`` JSON record to an :class:`ActiveMarket`.

    Returns ``None`` if the record is missing required fields.
    """
    resolved_spec = spec or DEFAULT_MARKET_DISCOVERY_SPEC
    slug = record.get("slug") or ""
    condition_id = record.get("conditionId") or record.get("condition_id") or ""
    raw_tokens = record.get("clobTokenIds")
    raw_outcomes = record.get("outcomes")

    # Gamma sometimes returns these as JSON-encoded strings.
    if isinstance(raw_tokens, str):
        try:
            raw_tokens = orjson.loads(raw_tokens)
        except orjson.JSONDecodeError:
            raw_tokens = None
    if isinstance(raw_outcomes, str):
        try:
            raw_outcomes = orjson.loads(raw_outcomes)
        except orjson.JSONDecodeError:
            raw_outcomes = None

    if not slug or not condition_id or not raw_tokens or not raw_outcomes:
        return None
    if len(raw_tokens) != 2 or len(raw_outcomes) != 2:
        return None

    # Outcomes for these markets are ["Up", "Down"]; we preserve that ordering.
    outcomes_normalised = [str(o).upper() for o in raw_outcomes]
    try:
        up_idx = outcomes_normalised.index("UP")
        down_idx = outcomes_normalised.index("DOWN")
    except ValueError:
        logger.warning(
            "gamma.unexpected_outcomes",
            extra={"slug": slug, "outcomes": raw_outcomes},
        )
        return None

    return ActiveMarket(
        slug=slug,
        condition_id=condition_id,
        asset_id_up=str(raw_tokens[up_idx]),
        asset_id_down=str(raw_tokens[down_idx]),
        start_ts_ms=_parse_iso8601_to_ms(record.get("startDate")),
        end_ts_ms=_parse_iso8601_to_ms(record.get("endDate")),
        tick_size=str(record.get("orderPriceMinTickSize", "0.01")),
        underlying=resolved_spec.underlying,
        horizon_ms=resolved_spec.horizon_ms,
        symbol_kind=resolved_spec.symbol_kind,
    )


class GammaClient:
    """Async REST client over ``gamma-api.polymarket.com``.

    Connection pool is owned per-instance; use as an async context manager.
    """

    def __init__(
        self,
        base_url: str,
        slug_prefix: str,
        *,
        request_timeout_seconds: float = 10.0,
        market_specs: Sequence[MarketDiscoverySpec] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._slug_prefix = slug_prefix
        self._market_specs = _normalise_market_specs(market_specs, slug_prefix=slug_prefix)
        self._timeout = aiohttp.ClientTimeout(total=request_timeout_seconds)
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> GammaClient:
        self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def list_active_markets(
        self,
        *,
        page_limit: int = _GAMMA_MAX_PAGE_LIMIT,
        max_pages: int = 60,
        empty_page_streak_limit: int = 3,
        page_concurrency: int = 8,
        direct_slug_lookback_intervals: int = 1,
        direct_slug_lookahead_intervals: int = 8,
        now_ms: int | None = None,
    ) -> list[ActiveMarket]:
        """Return all currently active markets whose slug starts with ``slug_prefix``.

        Iterates Gamma's offset-paginated ``/markets`` endpoint. We sort by
        ``endDate`` ascending so the nearest unresolved short-horizon markets
        surface first. Polymarket can create these markets many hours before
        their actual 15-minute settlement window, so creation/start-date
        ordering can otherwise over-subscribe future rounds while missing the
        live round.

        Gamma currently caps ``limit`` at 100 even if callers ask for more, so
        we cap the request size locally and advance by the number of records
        actually returned. That keeps us from skipping every other server page.

        Stops paginating when any of these is true:
        - a page is shorter than the effective request limit (end of stream), or
        - ``max_pages`` (safety cap).

        ``empty_page_streak_limit`` is retained for API compatibility, but the
        scan no longer stops on empty non-target pages. BTC 15m markets can be
        separated by unrelated market pages; stopping early caused the runner
        to drop current subscriptions.

        Records whose ``endDate`` is in the past or which fail validation are
        filtered out.
        """
        _ = empty_page_streak_limit
        assert self._session is not None, "use as async context manager"
        url = f"{self._base_url}/markets"
        out: list[ActiveMarket] = []
        request_limit = min(page_limit, _GAMMA_MAX_PAGE_LIMIT)
        resolved_now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
        seen_conditions: set[str] = set()

        try:
            for spec in self._market_specs:
                direct_records = await self._fetch_candidate_slug_markets(
                    url,
                    spec=spec,
                    now_ms=resolved_now_ms,
                    lookback_intervals=direct_slug_lookback_intervals,
                    lookahead_intervals=direct_slug_lookahead_intervals,
                )
                for rec in direct_records:
                    market = _market_from_gamma(dict(rec), spec)
                    if market is None or market.condition_id in seen_conditions:
                        continue
                    if market.end_ts_ms and market.end_ts_ms <= resolved_now_ms:
                        continue
                    seen_conditions.add(market.condition_id)
                    out.append(market)

            first_page = await self._fetch_markets_page(url, limit=request_limit, offset=0)
            pages: list[tuple[int, Sequence[Mapping[str, Any]]]] = [(0, first_page)]
            if len(first_page) >= request_limit and max_pages > 1:
                semaphore = asyncio.Semaphore(max(1, page_concurrency))

                async def fetch_offset(
                    page_index: int,
                ) -> tuple[int, Sequence[Mapping[str, Any]]]:
                    offset = page_index * request_limit
                    async with semaphore:
                        records = await self._fetch_markets_page(
                            url,
                            limit=request_limit,
                            offset=offset,
                        )
                    return offset, records

                pages.extend(
                    await asyncio.gather(
                        *(fetch_offset(page_index) for page_index in range(1, max_pages))
                    )
                )
        except (aiohttp.ClientError, TimeoutError):
            GAMMA_POLLS_TOTAL.labels(outcome="error").inc()
            raise

        for _, records in sorted(pages, key=lambda item: item[0]):
            if not isinstance(records, list):
                break
            for rec in records:
                slug = rec.get("slug") or ""
                spec = self._spec_for_slug(str(slug))
                if spec is None:
                    continue
                market = _market_from_gamma(dict(rec), spec)
                if market is None:
                    continue
                if market.condition_id in seen_conditions:
                    continue
                if market.end_ts_ms and market.end_ts_ms <= resolved_now_ms:
                    # Already resolved; Gamma's active=true filter occasionally
                    # lags real-time resolution.
                    continue
                seen_conditions.add(market.condition_id)
                out.append(market)

            if not records or len(records) < request_limit:
                break

        GAMMA_POLLS_TOTAL.labels(outcome="ok").inc()
        return out

    def _spec_for_slug(self, slug: str) -> MarketDiscoverySpec | None:
        for spec in self._market_specs:
            if slug.startswith(spec.slug_prefix):
                return spec
        return None

    async def _fetch_markets_page(
        self,
        url: str,
        *,
        limit: int,
        offset: int,
    ) -> list[Mapping[str, Any]]:
        assert self._session is not None, "use as async context manager"
        params = {
            "active": "true",
            "closed": "false",
            "limit": limit,
            "offset": offset,
            "order": "endDate",
            "ascending": "true",
        }
        try:
            async with self._session.get(url, params=params) as resp:
                resp.raise_for_status()
                raw = await resp.read()
        except (aiohttp.ClientError, TimeoutError) as exc:
            _log_gamma_fetch_failed(
                exc,
                url=url,
                limit=limit,
                offset=offset,
                timeout_s=self._timeout.total,
            )
            raise
        records = orjson.loads(raw)
        return records if isinstance(records, list) else []

    async def _fetch_candidate_slug_markets(
        self,
        url: str,
        *,
        spec: MarketDiscoverySpec,
        now_ms: int,
        lookback_intervals: int,
        lookahead_intervals: int,
    ) -> list[Mapping[str, Any]]:
        slugs = _candidate_slugs_for_now(
            spec.slug_prefix,
            now_ms=now_ms,
            horizon_ms=spec.horizon_ms,
            lookback_intervals=lookback_intervals,
            lookahead_intervals=lookahead_intervals,
        )
        if not slugs:
            return []

        async def fetch_one(slug: str) -> Mapping[str, Any] | None:
            return await self._fetch_market_by_slug(url, slug)

        records = await asyncio.gather(*(fetch_one(slug) for slug in slugs))
        return [record for record in records if record is not None]

    async def _fetch_market_by_slug(
        self,
        url: str,
        slug: str,
    ) -> Mapping[str, Any] | None:
        assert self._session is not None, "use as async context manager"
        params = {
            "active": "true",
            "closed": "false",
            "limit": 1,
            "slug": slug,
        }
        try:
            async with self._session.get(url, params=params) as resp:
                resp.raise_for_status()
                raw = await resp.read()
        except (aiohttp.ClientError, TimeoutError) as exc:
            _log_gamma_fetch_failed(
                exc,
                url=url,
                limit=1,
                offset=0,
                timeout_s=self._timeout.total,
            )
            raise
        records = orjson.loads(raw)
        if isinstance(records, list) and records and isinstance(records[0], Mapping):
            return records[0]
        return None


def diff_subscription_sets(
    current: Iterable[str],
    desired: Iterable[str],
) -> tuple[list[str], list[str]]:
    """Return ``(to_subscribe, to_unsubscribe)`` asset_id lists.

    Pure helper for the subscription manager; trivially unit-testable.
    """
    current_set = set(current)
    desired_set = set(desired)
    to_subscribe = sorted(desired_set - current_set)
    to_unsubscribe = sorted(current_set - desired_set)
    return to_subscribe, to_unsubscribe


def _candidate_slugs_for_now(
    slug_prefix: str,
    *,
    now_ms: int,
    horizon_ms: int,
    lookback_intervals: int,
    lookahead_intervals: int,
) -> tuple[str, ...]:
    if horizon_ms <= 0:
        return ()
    current_start = (now_ms // horizon_ms) * horizon_ms
    starts = [
        current_start + offset * horizon_ms
        for offset in range(-max(0, lookback_intervals), max(0, lookahead_intervals) + 1)
    ]
    return tuple(f"{slug_prefix}{start // 1000}" for start in starts if start >= 0)


def active_market_symbol_mapping_rows(
    markets: Sequence[ActiveMarket],
    *,
    ingest_ts: int | None = None,
) -> list[dict[str, Any]]:
    """Return canonical symbol mapping rows for active up/down markets."""

    resolved_ingest_ts = int(time.time() * 1000) if ingest_ts is None else int(ingest_ts)
    rows: list[dict[str, Any]] = []
    for market in markets:
        effective_from_ts = market.start_ts_ms if market.start_ts_ms > 0 else resolved_ingest_ts
        rows.append(
            _active_market_symbol_mapping_row(
                market,
                outcome_side="UP",
                source_symbol=market.asset_id_up,
                effective_from_ts=effective_from_ts,
                ingest_ts=resolved_ingest_ts,
            )
        )
        rows.append(
            _active_market_symbol_mapping_row(
                market,
                outcome_side="DOWN",
                source_symbol=market.asset_id_down,
                effective_from_ts=effective_from_ts,
                ingest_ts=resolved_ingest_ts,
            )
        )
    return rows


def _active_market_symbol_mapping_row(
    market: ActiveMarket,
    *,
    outcome_side: str,
    source_symbol: str,
    effective_from_ts: int,
    ingest_ts: int,
) -> dict[str, Any]:
    canonical_symbol = (
        f"{market.underlying}-{_horizon_label(market.horizon_ms)}:"
        f"{market.slug}:{outcome_side}"
    )
    return symbol_mapping_row(
        source=POLYMARKET_SOURCE,
        source_symbol=source_symbol,
        source_market=market.condition_id,
        canonical_symbol=canonical_symbol,
        effective_from_ts=effective_from_ts,
        ingest_ts=ingest_ts,
        message_ts=ingest_ts,
        symbol_kind=market.symbol_kind,
        metadata={
            "slug": market.slug,
            "outcome_side": outcome_side,
            "condition_id": market.condition_id,
            "start_ts_ms": market.start_ts_ms,
            "end_ts_ms": market.end_ts_ms,
            "tick_size": market.tick_size,
            "underlying": market.underlying,
            "horizon_ms": market.horizon_ms,
        },
    )


def parse_market_specs_json(
    raw: str | None,
    *,
    fallback_slug_prefix: str = "btc-updown-15m-",
) -> tuple[MarketDiscoverySpec, ...]:
    """Parse ``BIGAN_MARKET_SPECS_JSON`` into market discovery specs.

    The expected shape is a JSON array, for example:
    ``[{"slug_prefix":"btc-updown-15m-","underlying":"BTC","horizon_minutes":15}]``.
    """

    if raw is None or not raw.strip():
        return _normalise_market_specs(None, slug_prefix=fallback_slug_prefix)
    try:
        payload = orjson.loads(raw)
    except orjson.JSONDecodeError as exc:
        raise ValueError("market_specs_json must be valid JSON") from exc
    if not isinstance(payload, list):
        raise ValueError("market_specs_json must be a JSON array")

    specs: list[MarketDiscoverySpec] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise ValueError(f"market spec {idx} must be an object")
        slug_prefix = _required_text(item.get("slug_prefix"), f"market spec {idx}.slug_prefix")
        underlying = _optional_text(item.get("underlying")) or _infer_underlying(slug_prefix)
        horizon_ms = _horizon_ms_from_spec(item, slug_prefix=slug_prefix)
        symbol_kind = (
            _optional_text(item.get("symbol_kind"))
            or f"{underlying.lower()}_{_horizon_label(horizon_ms).lower()}_outcome"
        )
        specs.append(
            MarketDiscoverySpec(
                slug_prefix=slug_prefix,
                underlying=underlying.upper(),
                horizon_ms=horizon_ms,
                symbol_kind=symbol_kind,
            )
        )
    return _normalise_market_specs(specs, slug_prefix=fallback_slug_prefix)


def _normalise_market_specs(
    market_specs: Sequence[MarketDiscoverySpec] | None,
    *,
    slug_prefix: str,
) -> tuple[MarketDiscoverySpec, ...]:
    specs = tuple(market_specs or ())
    if not specs:
        horizon_ms = _infer_horizon_ms(slug_prefix) or BTC_15M_HORIZON_MS
        underlying = _infer_underlying(slug_prefix)
        return (
            MarketDiscoverySpec(
                slug_prefix=slug_prefix,
                underlying=underlying,
                horizon_ms=horizon_ms,
                symbol_kind=f"{underlying.lower()}_{_horizon_label(horizon_ms).lower()}_outcome",
            ),
        )
    deduped: list[MarketDiscoverySpec] = []
    seen: set[str] = set()
    for spec in specs:
        if not spec.slug_prefix:
            raise ValueError("market discovery slug_prefix is required")
        if spec.horizon_ms <= 0:
            raise ValueError("market discovery horizon_ms must be positive")
        if spec.slug_prefix in seen:
            continue
        seen.add(spec.slug_prefix)
        deduped.append(spec)
    return tuple(deduped)


def _horizon_ms_from_spec(item: Mapping[str, Any], *, slug_prefix: str) -> int:
    raw_ms = item.get("horizon_ms")
    if raw_ms is not None:
        return _positive_int(raw_ms, "horizon_ms")
    raw_minutes = item.get("horizon_minutes")
    if raw_minutes is not None:
        return _positive_int(raw_minutes, "horizon_minutes") * 60_000
    inferred = _infer_horizon_ms(slug_prefix)
    if inferred is None:
        raise ValueError("market spec requires horizon_ms or horizon_minutes")
    return inferred


def _infer_horizon_ms(slug_prefix: str) -> int | None:
    for part in slug_prefix.rstrip("-").split("-"):
        if len(part) < 2:
            continue
        value_text = part[:-1]
        suffix = part[-1].lower()
        if not value_text.isdigit():
            continue
        value = int(value_text)
        if suffix == "m":
            return value * 60_000
        if suffix == "h":
            return value * 60 * 60_000
    return None


def _infer_underlying(slug_prefix: str) -> str:
    head = slug_prefix.split("-", 1)[0].strip().upper()
    return head or "UNKNOWN"


def _horizon_label(horizon_ms: int) -> str:
    minutes = horizon_ms // 60_000
    if minutes > 0 and minutes * 60_000 == horizon_ms:
        return f"{minutes}M"
    return f"{horizon_ms}MS"


def _required_text(value: Any, name: str) -> str:
    text = _optional_text(value)
    if not text:
        raise ValueError(f"{name} is required")
    return text


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _positive_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _log_gamma_fetch_failed(
    exc: BaseException,
    *,
    url: str,
    limit: int,
    offset: int,
    timeout_s: float | None,
) -> None:
    context = _gamma_fetch_error_context(
        exc,
        url=url,
        limit=limit,
        offset=offset,
        timeout_s=timeout_s,
    )
    logger.warning(
        (
            "gamma.poll_failed err_type=%s err=%r url=%s limit=%s offset=%s "
            "timeout_s=%s cause_type=%s cause=%r"
        ),
        context["err_type"],
        context["err"],
        context["url"],
        context["limit"],
        context["offset"],
        context["timeout_s"],
        context["cause_type"],
        context["cause"],
        extra=context,
    )


def _gamma_fetch_error_context(
    exc: BaseException,
    *,
    url: str,
    limit: int,
    offset: int,
    timeout_s: float | None,
) -> dict[str, object]:
    """Return Gamma poll diagnostics that survive plain logging formatters."""

    cause = exc.__cause__
    return {
        "err_type": type(exc).__name__,
        "err": str(exc),
        "url": url,
        "limit": limit,
        "offset": offset,
        "timeout_s": timeout_s,
        "cause_type": type(cause).__name__ if cause is not None else None,
        "cause": str(cause) if cause is not None else None,
    }
