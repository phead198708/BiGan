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
POLYMARKET_SOURCE = "polymarket"
BTC_15M_SYMBOL_KIND = "btc_15m_outcome"


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


def _market_from_gamma(record: dict[str, Any]) -> ActiveMarket | None:
    """Convert one Gamma ``/markets`` JSON record to an :class:`ActiveMarket`.

    Returns ``None`` if the record is missing required fields.
    """
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
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._slug_prefix = slug_prefix
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
    ) -> list[ActiveMarket]:
        """Return all currently active markets whose slug starts with ``slug_prefix``.

        Iterates Gamma's offset-paginated ``/markets`` endpoint. We sort by
        ``startDate`` *descending* so the latest short-horizon markets surface
        first: at any moment the active set is dominated by a few hundred
        long-running markets, and the freshly-opened ``btc-updown-15m-*``
        markets cluster at the top of the newest-first ordering.

        Gamma currently caps ``limit`` at 100 even if callers ask for more, so
        we cap the request size locally and advance by the number of records
        actually returned. That keeps us from skipping every other server page.

        Stops paginating when any of these is true:
        - a page is shorter than the effective request limit (end of stream), or
        - ``max_pages`` (safety cap).

        ``empty_page_streak_limit`` is retained for API compatibility, but the
        scan no longer stops on empty non-target pages. Gamma sorts by market
        creation time, and BTC 15m markets can be separated by unrelated market
        pages; stopping early caused the runner to drop current subscriptions
        when a future BTC cohort appeared near the top.

        Records whose ``endDate`` is in the past or which fail validation are
        filtered out.
        """
        _ = empty_page_streak_limit
        assert self._session is not None, "use as async context manager"
        url = f"{self._base_url}/markets"
        out: list[ActiveMarket] = []
        request_limit = min(page_limit, _GAMMA_MAX_PAGE_LIMIT)
        now_ms = int(time.time() * 1000)
        seen_conditions: set[str] = set()

        try:
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
                if not slug.startswith(self._slug_prefix):
                    continue
                market = _market_from_gamma(rec)
                if market is None:
                    continue
                if market.condition_id in seen_conditions:
                    continue
                if market.end_ts_ms and market.end_ts_ms < now_ms:
                    # Already resolved; Gamma's active=true filter occasionally
                    # lags real-time resolution.
                    continue
                seen_conditions.add(market.condition_id)
                out.append(market)

            if not records or len(records) < request_limit:
                break

        GAMMA_POLLS_TOTAL.labels(outcome="ok").inc()
        return out

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
            "order": "startDate",
            "ascending": "false",
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


def active_market_symbol_mapping_rows(
    markets: Sequence[ActiveMarket],
    *,
    ingest_ts: int | None = None,
) -> list[dict[str, Any]]:
    """Return canonical symbol mapping rows for active BTC 15-minute markets."""

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
    canonical_symbol = f"BTC-15M:{market.slug}:{outcome_side}"
    return symbol_mapping_row(
        source=POLYMARKET_SOURCE,
        source_symbol=source_symbol,
        source_market=market.condition_id,
        canonical_symbol=canonical_symbol,
        effective_from_ts=effective_from_ts,
        ingest_ts=ingest_ts,
        message_ts=ingest_ts,
        symbol_kind=BTC_15M_SYMBOL_KIND,
        metadata={
            "slug": market.slug,
            "outcome_side": outcome_side,
            "condition_id": market.condition_id,
            "start_ts_ms": market.start_ts_ms,
            "end_ts_ms": market.end_ts_ms,
            "tick_size": market.tick_size,
        },
    )


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
