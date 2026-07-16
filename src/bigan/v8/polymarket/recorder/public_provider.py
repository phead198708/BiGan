"""Read-only provider contracts for real Polymarket corpus recording."""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import orjson
import websockets

from bigan.ingestion.message_types import (
    BestBidAskEvent,
    BookEvent,
    MarketResolvedEvent,
    PriceChangeEvent,
    UnknownEvent,
    parse_event,
)
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.corpus import BTC_UPDOWN_MARKET_HORIZONS_MS
from bigan.v8.polymarket.corpus.contracts import safety_fields
from bigan.v8.polymarket.recorder.contracts import PolymarketRealCorpusRecorderConfig
from bigan.v8.polymarket.recorder.market_identity_cache import (
    GAMMA_MARKET_IDENTITY_CACHE_FALLBACK_SOURCE_TYPE,
    GammaMarketIdentityCache,
    GammaMarketIdentityCacheError,
)

BTC_UPDOWN_SLUG_PATTERN = re.compile(r"^btc-updown-(5m|15m|1h)-(\d+)$")
BTC_UPDOWN_FAMILY_BY_SLUG = {
    "5m": "btc_updown_5m",
    "15m": "btc_updown_15m",
    "1h": "btc_updown_1h",
}
BTC_UPDOWN_SLUG_HORIZON_BY_FAMILY = {
    "btc_updown_5m": "5m",
    "btc_updown_15m": "15m",
    "btc_updown_1h": "1h",
}
DEFAULT_POLYMARKET_CLOB_WS_MARKET_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
POLYMARKET_CLOB_WS_SOURCE_CHANNEL = "polymarket_clob_ws_market"
BTC_FEATURE_SOURCE_COINBASE = "coinbase_btc_usd"
BTC_FEATURE_SOURCE_KRAKEN = "kraken_xbt_usd"
BTC_FEATURE_SOURCE_BINANCE = "binance_btcusdt"
DEFAULT_BTC_FEATURE_SOURCE_ORDER = ("coinbase", "kraken", "binance")
GAMMA_PRIMARY_MARKET_IDENTITY_SOURCE_TYPE = "gamma_primary"
_ACTIVE_GAMMA_PREFETCH_CACHE_PATHS: set[str] = set()
_ACTIVE_GAMMA_PREFETCH_LOCK = threading.Lock()


class RealCorpusPublicProviderError(RuntimeError):
    """Raised when a public read-only provider cannot safely normalize rows."""

    def __init__(self, message: str, *, reason_codes: tuple[str, ...]) -> None:
        super().__init__(message)
        self.reason_codes = reason_codes


class PolymarketOrderBookSource(Protocol):
    """Read-only source for CLOB book snapshots keyed by token id."""

    def book_payloads(self, token_ids: tuple[str, ...]) -> dict[str, dict[str, Any]]:
        """Return CLOB-like book payloads keyed by token id."""


class PolymarketRealCorpusPublicProvider(Protocol):
    """Normalized read-only public-data provider for the recorder operator.

    Implementations must return rows already normalized to the recorder raw
    contracts. The operator still validates every row and fails closed on
    provider exceptions or unsafe provider flags.
    """

    read_only: bool
    write_capable: bool
    paper_only: bool
    capital_at_risk: bool
    broker_exchange_write_enabled: bool
    live_exchange_write_enabled: bool
    polymarket_write_enabled: bool
    wallet_signing_enabled: bool

    def market_rows(
        self,
        config: PolymarketRealCorpusRecorderConfig,
    ) -> list[dict[str, Any]]:
        """Return normalized Polymarket Gamma market metadata rows."""

    def orderbook_rows(
        self,
        markets: list[dict[str, Any]],
        config: PolymarketRealCorpusRecorderConfig,
    ) -> list[dict[str, Any]]:
        """Return normalized Polymarket CLOB orderbook rows."""

    def trade_rows(
        self,
        markets: list[dict[str, Any]],
        config: PolymarketRealCorpusRecorderConfig,
    ) -> list[dict[str, Any]]:
        """Return normalized Polymarket CLOB trade rows."""

    def btc_feature_candle_rows(
        self,
        markets: list[dict[str, Any]],
        config: PolymarketRealCorpusRecorderConfig,
    ) -> list[dict[str, Any]]:
        """Return normalized causal BTC feature candle rows."""

    def resolution_rows(
        self,
        markets: list[dict[str, Any]],
        config: PolymarketRealCorpusRecorderConfig,
    ) -> list[dict[str, Any]]:
        """Return normalized official Polymarket resolution/reference rows."""


class PolymarketPublicHTTPRealCorpusProvider:
    """Read-only public provider for Polymarket BTC UP/DOWN corpus facts.

    This provider reads Gamma/Data API/CLOB websocket/BTC reference endpoints
    and normalizes what those endpoints actually expose. It does not synthesize
    historical bid/ask orderbooks from price history, and it does not use BTC
    feature candles as official Polymarket settlement evidence.
    """

    read_only = True
    write_capable = False
    paper_only = True
    capital_at_risk = False
    broker_exchange_write_enabled = False
    live_exchange_write_enabled = False
    polymarket_write_enabled = False
    wallet_signing_enabled = False

    def __init__(
        self,
        *,
        market_slugs: tuple[str, ...] = (),
        gamma_markets_endpoint: str = "https://gamma-api.polymarket.com/markets",
        gamma_events_endpoint: str = "https://gamma-api.polymarket.com/events/slug",
        clob_book_endpoint: str = "https://clob.polymarket.com/book",
        clob_books_endpoint: str = "https://clob.polymarket.com/books",
        clob_market_endpoint: str = "https://clob.polymarket.com/markets",
        clob_ws_url: str = DEFAULT_POLYMARKET_CLOB_WS_MARKET_URL,
        data_trades_endpoint: str = "https://data-api.polymarket.com/trades",
        coinbase_candles_endpoint: str = "https://api.exchange.coinbase.com/products/BTC-USD/candles",
        kraken_ohlc_endpoint: str = "https://api.kraken.com/0/public/OHLC",
        binance_klines_endpoint: str = "https://api.binance.com/api/v3/klines",
        btc_feature_source_order: tuple[str, ...] = DEFAULT_BTC_FEATURE_SOURCE_ORDER,
        max_markets: int = 3,
        recent_trade_limit: int = 250,
        timeout_seconds: float = 15.0,
        http_timeout_seconds: float | None = None,
        orderbook_snapshot_interval_seconds: float = 1.0,
        orderbook_ws_initial_complete_book_timeout_seconds: float = 15.0,
        rest_fallback_collection_seconds: float = 0.0,
        seed_rest_orderbooks_before_stream: bool = False,
        current_time_ms: int | None = None,
        fetch_json: Callable[[str], Any] | None = None,
        orderbook_source: PolymarketOrderBookSource | None = None,
        use_rest_orderbooks: bool = False,
        market_identity_cache_path: Path | str | None = None,
        market_identity_cache_max_age_seconds: float = 7_200.0,
        gamma_market_identity_prefetch_round_count: int = 0,
        clob_identity_revalidation_max_attempts: int = 3,
        clob_identity_revalidation_retry_seconds: float = 0.25,
    ) -> None:
        if max_markets <= 0:
            raise ValueError("max_markets must be positive")
        if recent_trade_limit <= 0:
            raise ValueError("recent_trade_limit must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if http_timeout_seconds is not None and http_timeout_seconds <= 0:
            raise ValueError("http_timeout_seconds must be positive")
        if orderbook_snapshot_interval_seconds <= 0:
            raise ValueError("orderbook_snapshot_interval_seconds must be positive")
        if orderbook_ws_initial_complete_book_timeout_seconds <= 0:
            raise ValueError(
                "orderbook_ws_initial_complete_book_timeout_seconds must be positive"
            )
        if rest_fallback_collection_seconds < 0:
            raise ValueError("rest_fallback_collection_seconds must be non-negative")
        if market_identity_cache_max_age_seconds <= 0:
            raise ValueError("market_identity_cache_max_age_seconds must be positive")
        if gamma_market_identity_prefetch_round_count < 0:
            raise ValueError(
                "gamma_market_identity_prefetch_round_count must be non-negative"
            )
        if clob_identity_revalidation_max_attempts <= 0:
            raise ValueError(
                "clob_identity_revalidation_max_attempts must be positive"
            )
        if clob_identity_revalidation_retry_seconds < 0:
            raise ValueError(
                "clob_identity_revalidation_retry_seconds must be non-negative"
            )
        if seed_rest_orderbooks_before_stream:
            raise ValueError(
                "pre-stream REST orderbook seeding is disabled; REST is fallback-only"
            )
        source_order = tuple(dict.fromkeys(source.strip().lower() for source in btc_feature_source_order))
        unsupported_sources = set(source_order) - {"coinbase", "kraken", "binance"}
        if unsupported_sources:
            raise ValueError(
                "unsupported BTC feature sources: " + ", ".join(sorted(unsupported_sources))
            )
        if not source_order:
            raise ValueError("btc_feature_source_order must not be empty")
        self.market_slugs = tuple(dict.fromkeys(slug.strip() for slug in market_slugs if slug.strip()))
        self.gamma_markets_endpoint = gamma_markets_endpoint
        self.gamma_events_endpoint = gamma_events_endpoint
        self.clob_book_endpoint = clob_book_endpoint
        self.clob_books_endpoint = clob_books_endpoint
        self.clob_market_endpoint = clob_market_endpoint
        self.clob_ws_url = clob_ws_url
        self.data_trades_endpoint = data_trades_endpoint
        self.coinbase_candles_endpoint = coinbase_candles_endpoint
        self.kraken_ohlc_endpoint = kraken_ohlc_endpoint
        self.binance_klines_endpoint = binance_klines_endpoint
        self.btc_feature_source_order = source_order
        self.max_markets = max_markets
        self.recent_trade_limit = recent_trade_limit
        self.timeout_seconds = timeout_seconds
        self.http_timeout_seconds = timeout_seconds if http_timeout_seconds is None else http_timeout_seconds
        self.orderbook_snapshot_interval_seconds = orderbook_snapshot_interval_seconds
        self.orderbook_ws_initial_complete_book_timeout_seconds = (
            orderbook_ws_initial_complete_book_timeout_seconds
        )
        self.rest_fallback_collection_seconds = rest_fallback_collection_seconds
        self.seed_rest_orderbooks_before_stream = seed_rest_orderbooks_before_stream
        self.current_time_ms = current_time_ms
        self._fetch_json = fetch_json
        self.gamma_market_identity_prefetch_round_count = (
            gamma_market_identity_prefetch_round_count
        )
        self.clob_identity_revalidation_max_attempts = (
            clob_identity_revalidation_max_attempts
        )
        self.clob_identity_revalidation_retry_seconds = (
            clob_identity_revalidation_retry_seconds
        )
        self._last_gamma_market_identity_prefetch_report: dict[str, Any] = {
            "prefetch_enabled": False,
            "requested_slug_count": 0,
            "stored_slug_count": 0,
            "reason_codes": ["gamma_market_identity_prefetch_not_run"],
            **safety_fields(),
        }
        self.market_identity_cache = (
            None
            if market_identity_cache_path is None
            else GammaMarketIdentityCache(
                market_identity_cache_path,
                max_age_seconds=market_identity_cache_max_age_seconds,
                current_time_ms=current_time_ms,
            )
        )
        if orderbook_source is not None:
            self.orderbook_source = orderbook_source
        elif use_rest_orderbooks:
            self.orderbook_source = None
        else:
            self.orderbook_source = PolymarketCLOBWebSocketOrderBookSource(
                ws_url=clob_ws_url,
                timeout_seconds=timeout_seconds,
                snapshot_interval_seconds=orderbook_snapshot_interval_seconds,
                initial_complete_book_timeout_seconds=(
                    orderbook_ws_initial_complete_book_timeout_seconds
                ),
            )

    def market_rows(
        self,
        config: PolymarketRealCorpusRecorderConfig,
    ) -> list[dict[str, Any]]:
        slugs = self.market_slugs or self._discover_current_btc_updown_slugs(config)
        rows: list[dict[str, Any]] = []
        primary_success = False
        for slug in slugs[: self.max_markets]:
            fetched_at_ts = self._current_time_ms()
            fallback_reason_codes: tuple[str, ...] = ()
            try:
                payloads = self._fetch_gamma_market_payloads(slug)
            except RealCorpusPublicProviderError as exc:
                if not _gamma_cache_fallback_allowed(exc.reason_codes):
                    raise
                payloads = []
                fallback_reason_codes = exc.reason_codes
            if payloads:
                normalized = [
                    row
                    for row in (
                        self._normalize_gamma_market_payload(payload, config)
                        for payload in payloads
                    )
                    if row is not None and row["slug"] == slug
                ]
                if len(normalized) > 1:
                    raise RealCorpusPublicProviderError(
                        "Gamma returned multiple identities for one exact slug.",
                        reason_codes=("gamma_market_identity_ambiguous",),
                    )
                if len(normalized) == 1:
                    primary_success = True
                    rows.extend(
                        _annotate_market_identity(
                            row,
                            source_type=GAMMA_PRIMARY_MARKET_IDENTITY_SOURCE_TYPE,
                            fetched_at_ts=fetched_at_ts,
                            cache_fallback=False,
                            fallback_reason_codes=(),
                            cache_entry_sha256=None,
                            cache_age_ms=None,
                            clob_validation=None,
                        )
                        for row in normalized
                    )
                    continue
                fallback_reason_codes = (
                    "real_public_collection_empty_market_discovery",
                )
            elif not fallback_reason_codes:
                fallback_reason_codes = (
                    "real_public_collection_empty_market_discovery",
                )
            rows.append(
                self._market_row_from_identity_cache(
                    slug=slug,
                    config=config,
                    decision_ts=self._current_time_ms(),
                    fallback_reason_codes=fallback_reason_codes,
                )
            )
        if not rows:
            raise RealCorpusPublicProviderError(
                "No BTC UP/DOWN Gamma markets could be normalized from public data.",
                reason_codes=("real_public_collection_empty_market_discovery",),
            )
        if primary_success:
            self._start_gamma_market_identity_prefetch(
                config=config,
                base_slugs=slugs,
            )
        return rows[: self.max_markets]

    def prefetch_gamma_market_identities(
        self,
        *,
        config: PolymarketRealCorpusRecorderConfig,
        base_slugs: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """Synchronously prefetch deterministic future slugs for diagnostics/tests."""

        cache = self.market_identity_cache
        requested_count = self.gamma_market_identity_prefetch_round_count
        if cache is None or requested_count <= 0:
            report = {
                "prefetch_enabled": False,
                "requested_slug_count": 0,
                "stored_slug_count": 0,
                "reason_codes": ["gamma_market_identity_prefetch_disabled"],
                **safety_fields(),
            }
            self._last_gamma_market_identity_prefetch_report = report
            return report
        current_slugs = base_slugs or self._discover_current_btc_updown_slugs(config)
        future_slugs = self._future_btc_updown_slugs(
            config=config,
            base_slugs=current_slugs,
            round_count=requested_count,
        )
        stored_slugs: list[str] = []
        reason_codes: list[str] = []
        for slug in future_slugs:
            try:
                payloads = self._fetch_gamma_market_payloads(slug)
            except RealCorpusPublicProviderError as exc:
                reason_codes.extend(exc.reason_codes)
                break
            if not payloads:
                reason_codes.append("gamma_market_identity_prefetch_empty")
                break
            fetched_at_ts = self._current_time_ms()
            normalized_rows = [
                row
                for row in (
                    self._normalize_gamma_market_payload(
                        payload,
                        config,
                        allow_future_market=True,
                    )
                    for payload in payloads
                )
                if row is not None and row["slug"] == slug
            ]
            if len(normalized_rows) != 1:
                reason_codes.append(
                    "gamma_market_identity_prefetch_not_exactly_one_market"
                )
                break
            row = normalized_rows[0]
            try:
                cache.store_prefetched_payload(
                    payload=dict(row["raw_public_payload"]),
                    slug=row["slug"],
                    market_family=row["market_family"],
                    market_start_ts=int(row["market_start_ts"]),
                    market_end_ts=int(row["market_end_ts"]),
                    condition_id=row["condition_id"],
                    up_token_id=row["up_token_id"],
                    down_token_id=row["down_token_id"],
                    reference_price_source=row["reference_price_source"],
                    settlement_rule=row["settlement_rule"],
                    fetched_at_ts=fetched_at_ts,
                    source_endpoint=self.gamma_markets_endpoint,
                )
            except GammaMarketIdentityCacheError as exc:
                reason_codes.extend(exc.reason_codes)
                break
            stored_slugs.append(slug)
        report = {
            "prefetch_enabled": True,
            "requested_slug_count": len(future_slugs),
            "stored_slug_count": len(stored_slugs),
            "stored_slugs": stored_slugs,
            "reason_codes": sorted(set(reason_codes)),
            "cache_report": cache.report(),
            **safety_fields(),
        }
        self._last_gamma_market_identity_prefetch_report = report
        return report

    def market_identity_cache_report(self) -> dict[str, Any]:
        if self.market_identity_cache is None:
            return {
                "cache_enabled": False,
                "cache_path": None,
                "cache_entry_count": 0,
                **safety_fields(),
            }
        report = self.market_identity_cache.report()
        report["cache_enabled"] = True
        report["last_prefetch_report"] = dict(
            self._last_gamma_market_identity_prefetch_report
        )
        return report

    def orderbook_rows(
        self,
        markets: list[dict[str, Any]],
        config: PolymarketRealCorpusRecorderConfig,
    ) -> list[dict[str, Any]]:
        if self.orderbook_source is not None:
            try:
                rows = self._orderbook_rows_from_stream_source(markets, config)
                if not rows:
                    rows = self._orderbook_rows_from_source(markets)
            except RealCorpusPublicProviderError as exc:
                return self._rest_orderbook_fallback(
                    markets,
                    reason_codes=exc.reason_codes,
                )

            rows = _annotate_orderbook_rows(
                rows,
                source_type="polymarket_clob_websocket",
                rest_fallback=False,
                fallback_reason_codes=(),
            )
            expected_token_ids = set(_token_ids_for_markets(markets))
            observed_token_ids = {str(row.get("token_id") or "") for row in rows}
            missing_token_ids = expected_token_ids - observed_token_ids
            if not missing_token_ids:
                return rows

            fallback_rows = self._rest_orderbook_fallback(
                markets,
                reason_codes=("polymarket_clob_ws_missing_token_orderbook",),
            )
            if self.rest_fallback_collection_seconds > 0:
                # Synchronized REST snapshots establish new complete paired
                # timestamps without backdating a missing token onto an
                # earlier partial WebSocket observation.
                rows.extend(fallback_rows)
            else:
                rows.extend(
                    row
                    for row in fallback_rows
                    if str(row.get("token_id") or "") in missing_token_ids
                )
            remaining_missing_token_ids = expected_token_ids - {
                str(row.get("token_id") or "") for row in rows
            }
            if remaining_missing_token_ids:
                raise RealCorpusPublicProviderError(
                    "WebSocket orderbook rows and REST fallback did not cover all tokens.",
                    reason_codes=(
                        "polymarket_clob_ws_and_rest_orderbook_coverage_incomplete",
                    ),
                )
            return _with_orderbook_collection_end(rows)
        return self._orderbook_rows_from_rest(markets)

    def _rest_orderbook_fallback(
        self,
        markets: list[dict[str, Any]],
        *,
        reason_codes: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        fallback_reason_codes = list(reason_codes)
        try:
            if self.rest_fallback_collection_seconds > 0:
                rows, collection_reason_codes = (
                    self._raw_orderbook_rows_from_rest_window(markets)
                )
                fallback_reason_codes.extend(collection_reason_codes)
            else:
                rows = self._raw_orderbook_rows_from_rest(markets)
        except RealCorpusPublicProviderError as exc:
            raise RealCorpusPublicProviderError(
                "WebSocket orderbook collection and REST fallback both failed.",
                reason_codes=tuple(
                    dict.fromkeys((*fallback_reason_codes, *exc.reason_codes))
                ),
            ) from exc
        if not rows:
            raise RealCorpusPublicProviderError(
                "WebSocket orderbook collection failed and REST fallback was empty.",
                reason_codes=tuple(
                    dict.fromkeys(
                        (
                            *fallback_reason_codes,
                            "polymarket_clob_rest_fallback_empty",
                        )
                    )
                ),
            )
        return _with_orderbook_collection_end(
            _annotate_orderbook_rows(
                rows,
                source_type="polymarket_clob_rest_fallback",
                rest_fallback=True,
                fallback_reason_codes=tuple(dict.fromkeys(fallback_reason_codes)),
            )
        )

    def _raw_orderbook_rows_from_rest_window(
        self,
        markets: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
        if not markets:
            return [], ()
        market_end_ts = min(int(market["market_end_ts"]) for market in markets)
        remaining_market_ms = market_end_ts - self._current_time_ms()
        if remaining_market_ms <= 0:
            raise RealCorpusPublicProviderError(
                "REST orderbook fallback cannot start after the market closed.",
                reason_codes=("polymarket_clob_rest_fallback_market_closed",),
            )
        collection_seconds = min(
            self.rest_fallback_collection_seconds,
            remaining_market_ms / 1000.0,
        )
        deadline = time.monotonic() + collection_seconds
        rows: list[dict[str, Any]] = []
        failure_reason_codes: list[str] = []
        snapshot_index = 0
        while True:
            snapshot_index += 1
            request_started_at_ts = self._current_time_ms()
            try:
                snapshot_rows = self._raw_orderbook_rows_from_rest(markets)
            except RealCorpusPublicProviderError as exc:
                failure_reason_codes.extend(exc.reason_codes)
                snapshot_rows = []
            observed_at_ts = self._current_time_ms()
            if observed_at_ts >= market_end_ts:
                failure_reason_codes.append(
                    "polymarket_clob_rest_fallback_snapshot_after_market_close"
                )
                break
            for row in snapshot_rows:
                available_at_ts = max(
                    int(row.get("available_at_ts") or row["ts"]),
                    observed_at_ts,
                )
                rows.append(
                    {
                        **row,
                        "available_at_ts": available_at_ts,
                        "rest_fallback_snapshot_index": snapshot_index,
                        "rest_fallback_request_started_at_ts": (
                            request_started_at_ts
                        ),
                        "rest_fallback_observed_at_ts": observed_at_ts,
                    }
                )
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                break
            time.sleep(
                min(
                    self.orderbook_snapshot_interval_seconds,
                    remaining_seconds,
                )
            )
        return rows, tuple(dict.fromkeys(failure_reason_codes))

    def _orderbook_rows_from_stream_source(
        self,
        markets: list[dict[str, Any]],
        config: PolymarketRealCorpusRecorderConfig,
    ) -> list[dict[str, Any]]:
        del config
        snapshot_getter = getattr(self.orderbook_source, "book_payload_snapshots", None)
        if not callable(snapshot_getter):
            return []
        rows: list[dict[str, Any]] = []
        snapshots = snapshot_getter(_token_ids_for_markets(markets))
        for payloads in snapshots:
            for market in markets:
                for outcome, token_id in (
                    ("UP", str(market["up_token_id"])),
                    ("DOWN", str(market["down_token_id"])),
                ):
                    payload = payloads.get(token_id)
                    if payload is None:
                        continue
                    row = self._normalize_book_payload(
                        market=market,
                        outcome=outcome,
                        token_id=token_id,
                        payload=payload,
                    )
                    if row is not None:
                        rows.append(row)
        return _with_orderbook_collection_end(rows)

    def _orderbook_rows_from_source(self, markets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        payloads = self.orderbook_source.book_payloads(_token_ids_for_markets(markets))
        rows: list[dict[str, Any]] = []
        for market in markets:
            for outcome, token_id in (
                ("UP", str(market["up_token_id"])),
                ("DOWN", str(market["down_token_id"])),
            ):
                payload = payloads.get(token_id)
                if payload is None:
                    continue
                row = self._normalize_book_payload(
                    market=market,
                    outcome=outcome,
                    token_id=token_id,
                    payload=payload,
                )
                if row is not None:
                    rows.append(row)
        return _with_orderbook_collection_end(rows)

    def _orderbook_rows_from_rest(self, markets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return _with_orderbook_collection_end(
            _annotate_orderbook_rows(
                self._raw_orderbook_rows_from_rest(markets),
                source_type="polymarket_clob_rest_explicit",
                rest_fallback=False,
                fallback_reason_codes=(),
            )
        )

    def _raw_orderbook_rows_from_rest(self, markets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        payloads = self._fetch_clob_books(_token_ids_for_markets(markets))
        for market in markets:
            for outcome, token_id in (
                ("UP", str(market["up_token_id"])),
                ("DOWN", str(market["down_token_id"])),
            ):
                payload = payloads.get(token_id)
                if payload is None:
                    payload = self._fetch_clob_book(token_id)
                row = self._normalize_book_payload(
                    market=market,
                    outcome=outcome,
                    token_id=token_id,
                    payload=payload,
                )
                if row is not None:
                    rows.append(row)
        return rows

    def trade_rows(
        self,
        markets: list[dict[str, Any]],
        config: PolymarketRealCorpusRecorderConfig,
    ) -> list[dict[str, Any]]:
        del config
        rows: list[dict[str, Any]] = []
        by_condition = {str(market["condition_id"]): market for market in markets}
        for condition_id, market in by_condition.items():
            params = urllib.parse.urlencode(
                {"market": condition_id, "limit": self.recent_trade_limit}
            )
            payload = self._get_json(f"{self.data_trades_endpoint}?{params}")
            trades = payload if isinstance(payload, list) else []
            for trade in trades:
                row = self._normalize_trade_payload(market=market, payload=trade)
                if row is not None:
                    rows.append(row)
        return rows

    def btc_feature_candle_rows(
        self,
        markets: list[dict[str, Any]],
        config: PolymarketRealCorpusRecorderConfig,
    ) -> list[dict[str, Any]]:
        if not markets:
            return []
        timeframe_ms = config.candle_timeframe_ms
        min_ts = min(int(market["market_start_ts"]) for market in markets) - 15 * 60_000
        max_ts = max(int(market["market_end_ts"]) for market in markets)
        collection_now_ms = self._current_time_ms()
        end_ts = min(max_ts + timeframe_ms, collection_now_ms)
        if end_ts <= min_ts:
            return []
        failures: list[str] = []
        for source in self.btc_feature_source_order:
            try:
                if source == "coinbase":
                    rows = self._coinbase_feature_candle_rows(
                        min_ts=min_ts,
                        end_ts=end_ts,
                        timeframe_ms=timeframe_ms,
                        collection_now_ms=collection_now_ms,
                    )
                elif source == "kraken":
                    rows = self._kraken_feature_candle_rows(
                        min_ts=min_ts,
                        end_ts=end_ts,
                        timeframe_ms=timeframe_ms,
                        collection_now_ms=collection_now_ms,
                    )
                else:
                    rows = self._binance_feature_candle_rows(
                        min_ts=min_ts,
                        end_ts=end_ts,
                        timeframe_ms=timeframe_ms,
                        collection_now_ms=collection_now_ms,
                    )
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{source}: {exc}")
                continue
            if rows:
                return rows
            failures.append(f"{source}: no usable closed candles")
        raise RealCorpusPublicProviderError(
            "No BTC feature candles were available from public sources: "
            + "; ".join(failures),
            reason_codes=("btc_feature_candle_sources_unavailable",),
        )

    def _coinbase_feature_candle_rows(
        self,
        *,
        min_ts: int,
        end_ts: int,
        timeframe_ms: int,
        collection_now_ms: int,
    ) -> list[dict[str, Any]]:
        params = urllib.parse.urlencode(
            {
                "start": _iso_millis(min_ts),
                "end": _iso_millis(end_ts),
                "granularity": timeframe_ms // 1000,
            }
        )
        payload = self._get_json(f"{self.coinbase_candles_endpoint}?{params}")
        if not isinstance(payload, list):
            raise RealCorpusPublicProviderError(
                "Invalid Coinbase candles public payload.",
                reason_codes=("invalid_btc_feature_candle_payload",),
            )
        return [
            row
            for row in (
                self._normalize_coinbase_candle(row, timeframe_ms, collection_now_ms)
                for row in payload
            )
            if row is not None
        ]

    def _kraken_feature_candle_rows(
        self,
        *,
        min_ts: int,
        end_ts: int,
        timeframe_ms: int,
        collection_now_ms: int,
    ) -> list[dict[str, Any]]:
        params = urllib.parse.urlencode(
            {
                "pair": "XBTUSD",
                "interval": _kraken_interval(timeframe_ms),
                "since": min_ts // 1000,
            }
        )
        payload = self._get_json(f"{self.kraken_ohlc_endpoint}?{params}")
        if not isinstance(payload, dict):
            raise RealCorpusPublicProviderError(
                "Invalid Kraken OHLC public payload.",
                reason_codes=("invalid_btc_feature_candle_payload",),
            )
        if payload.get("error"):
            raise RealCorpusPublicProviderError(
                "Kraken OHLC public payload returned errors: " + str(payload.get("error")),
                reason_codes=("invalid_btc_feature_candle_payload",),
            )
        rows = _kraken_ohlc_rows(payload)
        return [
            row
            for row in (
                self._normalize_kraken_ohlc(row, timeframe_ms, collection_now_ms)
                for row in rows
            )
            if row is not None and int(row["ts"]) < end_ts
        ]

    def _binance_feature_candle_rows(
        self,
        *,
        min_ts: int,
        end_ts: int,
        timeframe_ms: int,
        collection_now_ms: int,
    ) -> list[dict[str, Any]]:
        interval = _binance_interval(timeframe_ms)
        params = urllib.parse.urlencode(
            {
                "symbol": "BTCUSDT",
                "interval": interval,
                "startTime": min_ts,
                "endTime": end_ts,
                "limit": 1000,
            }
        )
        payload = self._get_json(f"{self.binance_klines_endpoint}?{params}")
        if not isinstance(payload, list):
            raise RealCorpusPublicProviderError(
                "Invalid Binance klines public payload.",
                reason_codes=("invalid_btc_feature_candle_payload",),
            )
        return [
            row
            for row in (
                self._normalize_binance_kline(row, timeframe_ms, collection_now_ms)
                for row in payload
            )
            if row is not None
        ]

    def resolution_rows(
        self,
        markets: list[dict[str, Any]],
        config: PolymarketRealCorpusRecorderConfig,
    ) -> list[dict[str, Any]]:
        del config
        rows = []
        ws_resolution_payloads = self._resolution_payloads_from_orderbook_source(markets)
        for market in markets:
            raw_payload = self._gamma_resolution_payload_for_market(market)
            event_payload = self._gamma_event_payload_for_market(market)
            reference_price_fields = _reference_price_fields_for_resolution(
                market=market,
                event_payload=event_payload,
                gamma_market_payload=raw_payload,
            )
            ws_resolution = _payout_resolution_from_market_resolved_payload(
                payload=ws_resolution_payloads.get(str(market["market_id"])),
                market=market,
            )
            if ws_resolution is not None:
                rows.append(
                    {
                        "market_id": market["market_id"],
                        "reference_price_source": market["reference_price_source"],
                        "resolution_status": ws_resolution["resolution_status"],
                        "resolved_outcome": ws_resolution["resolved_outcome"],
                        "payout_up": ws_resolution["payout_up"],
                        "payout_down": ws_resolution["payout_down"],
                        "resolution_source_type": "polymarket_clob_ws_market_resolved",
                        "raw_resolution_text": str(ws_resolution_payloads.get(str(market["market_id"]))),
                        **reference_price_fields,
                        **safety_fields(),
                    }
                )
                continue
            clob_market_payload = self._clob_market_payload_for_market(market)
            clob_resolution = _payout_resolution_from_clob_market_payload(
                payload=clob_market_payload,
                market=market,
            )
            if clob_resolution is not None:
                rows.append(
                    {
                        "market_id": market["market_id"],
                        "reference_price_source": market["reference_price_source"],
                        "resolution_status": clob_resolution["resolution_status"],
                        "resolved_outcome": clob_resolution["resolved_outcome"],
                        "payout_up": clob_resolution["payout_up"],
                        "payout_down": clob_resolution["payout_down"],
                        "resolution_source_type": "polymarket_clob_market_tokens",
                        "raw_resolution_text": str(clob_market_payload),
                        **reference_price_fields,
                        **safety_fields(),
                    }
                )
                continue
            if not reference_price_fields:
                payout_resolution = _payout_resolution_from_gamma_payload(
                    payload=raw_payload,
                    market=market,
                    current_time_ms=self._current_time_ms(),
                )
                if payout_resolution is None:
                    continue
                rows.append(
                    {
                        "market_id": market["market_id"],
                        "reference_price_source": market["reference_price_source"],
                        "resolution_status": payout_resolution["resolution_status"],
                        "resolved_outcome": payout_resolution["resolved_outcome"],
                        "payout_up": payout_resolution["payout_up"],
                        "payout_down": payout_resolution["payout_down"],
                        "resolution_source_type": "gamma_outcome_prices",
                        "raw_resolution_text": str(raw_payload.get("description") or ""),
                        **safety_fields(),
                    }
                )
            else:
                rows.append(
                    {
                        "market_id": market["market_id"],
                        "reference_price_source": market["reference_price_source"],
                        "resolution_status": _resolution_status_from_payload(raw_payload),
                        "resolution_source_type": "reference_prices",
                        "raw_resolution_text": str(raw_payload.get("description") or ""),
                        **reference_price_fields,
                        **safety_fields(),
                    }
                )
        return rows

    def _resolution_payloads_from_orderbook_source(
        self,
        markets: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        if self.orderbook_source is None:
            return {}
        payload_getter = getattr(self.orderbook_source, "market_resolution_payloads", None)
        if not callable(payload_getter):
            return {}
        try:
            payloads = payload_getter(_token_ids_for_markets(markets))
        except RealCorpusPublicProviderError:
            return {}
        return {str(market_id): dict(payload) for market_id, payload in payloads.items()}

    def _clob_market_payload_for_market(self, market: dict[str, Any]) -> dict[str, Any]:
        condition_id = str(market.get("condition_id") or market.get("market_id") or "")
        if not condition_id:
            return {}
        try:
            payload = self._get_json(f"{self.clob_market_endpoint}/{condition_id}")
        except RealCorpusPublicProviderError:
            return {}
        except Exception:
            return {}
        return dict(payload) if isinstance(payload, dict) else {}

    def _gamma_resolution_payload_for_market(self, market: dict[str, Any]) -> dict[str, Any]:
        base_payload = dict(market.get("raw_public_payload") or {})
        slug = str(market.get("slug") or "")
        if not slug:
            return base_payload
        try:
            payloads = self._fetch_gamma_market_payloads(slug)
        except RealCorpusPublicProviderError:
            return base_payload
        except Exception:
            return base_payload
        condition_id = str(market.get("condition_id") or market.get("market_id") or "")
        for payload in payloads:
            payload_slug = str(payload.get("slug") or payload.get("market_slug") or "")
            payload_condition = str(
                payload.get("conditionId") or payload.get("condition_id") or ""
            )
            if payload_slug != slug:
                continue
            if condition_id and payload_condition and payload_condition != condition_id:
                continue
            merged = dict(base_payload)
            merged.update(payload)
            return merged
        return base_payload

    def _gamma_event_payload_for_market(self, market: dict[str, Any]) -> dict[str, Any]:
        slug = str(market.get("slug") or "")
        if not slug:
            return {}
        try:
            payload = self._get_json(
                f"{self.gamma_events_endpoint.rstrip('/')}/{urllib.parse.quote(slug, safe='')}"
            )
        except RealCorpusPublicProviderError:
            return {}
        except Exception:
            return {}
        return dict(payload) if isinstance(payload, dict) else {}

    def _discover_current_btc_updown_slugs(
        self,
        config: PolymarketRealCorpusRecorderConfig,
    ) -> tuple[str, ...]:
        now_ms = self._current_time_ms()
        slugs: list[str] = []
        for family in config.market_families:
            horizon_name = BTC_UPDOWN_SLUG_HORIZON_BY_FAMILY[str(family)]
            horizon_ms = BTC_UPDOWN_MARKET_HORIZONS_MS[str(family)]
            start_epoch_seconds = (now_ms // horizon_ms) * horizon_ms // 1000
            slugs.append(f"btc-updown-{horizon_name}-{start_epoch_seconds}")
        return tuple(slugs)

    def _future_btc_updown_slugs(
        self,
        *,
        config: PolymarketRealCorpusRecorderConfig,
        base_slugs: tuple[str, ...],
        round_count: int,
    ) -> tuple[str, ...]:
        family_by_base_slug: dict[str, str] = {}
        for slug in base_slugs:
            match = BTC_UPDOWN_SLUG_PATTERN.match(slug)
            if match is None:
                continue
            family = BTC_UPDOWN_FAMILY_BY_SLUG[match.group(1)]
            if family in config.market_families:
                family_by_base_slug[slug] = family
        rows: list[str] = []
        for slug, family in family_by_base_slug.items():
            match = BTC_UPDOWN_SLUG_PATTERN.match(slug)
            if match is None:
                continue
            start_seconds = int(match.group(2))
            horizon_seconds = BTC_UPDOWN_MARKET_HORIZONS_MS[family] // 1000
            horizon_name = BTC_UPDOWN_SLUG_HORIZON_BY_FAMILY[family]
            rows.extend(
                f"btc-updown-{horizon_name}-{start_seconds + offset * horizon_seconds}"
                for offset in range(1, round_count + 1)
            )
        return tuple(dict.fromkeys(rows))

    def _start_gamma_market_identity_prefetch(
        self,
        *,
        config: PolymarketRealCorpusRecorderConfig,
        base_slugs: tuple[str, ...],
    ) -> None:
        cache = self.market_identity_cache
        if (
            cache is None
            or self.gamma_market_identity_prefetch_round_count <= 0
            or self.market_slugs
        ):
            return
        cache_key = str(cache.path)
        with _ACTIVE_GAMMA_PREFETCH_LOCK:
            if cache_key in _ACTIVE_GAMMA_PREFETCH_CACHE_PATHS:
                return
            _ACTIVE_GAMMA_PREFETCH_CACHE_PATHS.add(cache_key)

        def worker() -> None:
            try:
                self.prefetch_gamma_market_identities(
                    config=config,
                    base_slugs=base_slugs,
                )
            except Exception as exc:  # noqa: BLE001
                self._last_gamma_market_identity_prefetch_report = {
                    "prefetch_enabled": True,
                    "requested_slug_count": (
                        self.gamma_market_identity_prefetch_round_count
                    ),
                    "stored_slug_count": 0,
                    "reason_codes": [
                        "gamma_market_identity_prefetch_unexpected_error"
                    ],
                    "details": str(exc),
                    **safety_fields(),
                }
            finally:
                with _ACTIVE_GAMMA_PREFETCH_LOCK:
                    _ACTIVE_GAMMA_PREFETCH_CACHE_PATHS.discard(cache_key)

        threading.Thread(
            target=worker,
            name="gamma-market-identity-prefetch",
            daemon=True,
        ).start()

    def _market_row_from_identity_cache(
        self,
        *,
        slug: str,
        config: PolymarketRealCorpusRecorderConfig,
        decision_ts: int,
        fallback_reason_codes: tuple[str, ...],
    ) -> dict[str, Any]:
        cache = self.market_identity_cache
        if cache is None:
            raise RealCorpusPublicProviderError(
                "Gamma market discovery failed and no identity cache is configured.",
                reason_codes=tuple(
                    dict.fromkeys(
                        (
                            *fallback_reason_codes,
                            "gamma_market_identity_cache_not_configured",
                        )
                    )
                ),
            )
        match = BTC_UPDOWN_SLUG_PATTERN.match(slug)
        if match is None:
            raise RealCorpusPublicProviderError(
                "Gamma cache fallback received an unsupported market slug.",
                reason_codes=("gamma_market_identity_cache_slug_mismatch",),
            )
        family = BTC_UPDOWN_FAMILY_BY_SLUG[match.group(1)]
        start_ts = int(match.group(2)) * 1000
        end_ts = start_ts + BTC_UPDOWN_MARKET_HORIZONS_MS[family]
        try:
            entry = cache.lookup(
                slug=slug,
                decision_ts=decision_ts,
                expected_market_family=family,
                expected_market_start_ts=start_ts,
                expected_market_end_ts=end_ts,
            )
        except GammaMarketIdentityCacheError as exc:
            raise RealCorpusPublicProviderError(
                "Gamma market discovery failed and cache fallback was rejected.",
                reason_codes=tuple(
                    dict.fromkeys((*fallback_reason_codes, *exc.reason_codes))
                ),
            ) from exc
        row = self._normalize_gamma_market_payload(
            dict(entry["identity_payload"]),
            config,
        )
        if row is None or row["slug"] != slug:
            raise RealCorpusPublicProviderError(
                "Gamma cache payload could not normalize to the current market.",
                reason_codes=tuple(
                    dict.fromkeys(
                        (
                            *fallback_reason_codes,
                            "gamma_market_identity_cache_normalization_failed",
                        )
                    )
                ),
            )
        cached_identity = (
            str(entry["condition_id"]),
            str(entry["up_token_id"]),
            str(entry["down_token_id"]),
            int(entry["market_start_ts"]),
            int(entry["market_end_ts"]),
        )
        normalized_identity = (
            str(row["condition_id"]),
            str(row["up_token_id"]),
            str(row["down_token_id"]),
            int(row["market_start_ts"]),
            int(row["market_end_ts"]),
        )
        if normalized_identity != cached_identity:
            raise RealCorpusPublicProviderError(
                "Gamma cache entry fields disagree with its raw payload.",
                reason_codes=(
                    "gamma_market_identity_cache_payload_identity_mismatch",
                ),
            )
        clob_validation = self._validate_cached_identity_against_clob(row)
        return _annotate_market_identity(
            row,
            source_type=GAMMA_MARKET_IDENTITY_CACHE_FALLBACK_SOURCE_TYPE,
            fetched_at_ts=int(entry["fetched_at_ts"]),
            cache_fallback=True,
            fallback_reason_codes=fallback_reason_codes,
            cache_entry_sha256=str(entry["cache_entry_sha256"]),
            cache_age_ms=int(entry["cache_age_ms"]),
            clob_validation=clob_validation,
        )

    def _validate_cached_identity_against_clob(
        self,
        market: dict[str, Any],
    ) -> dict[str, Any]:
        condition_id = str(market["condition_id"])
        payload: Any = None
        retry_reason_codes: list[str] = []
        attempt_count = 0
        for attempt_count in range(
            1,
            self.clob_identity_revalidation_max_attempts + 1,
        ):
            try:
                payload = self._get_json(
                    f"{self.clob_market_endpoint}/{condition_id}"
                )
                break
            except RealCorpusPublicProviderError as exc:
                retry_reason_codes.extend(exc.reason_codes)
                if (
                    not _gamma_cache_fallback_allowed(exc.reason_codes)
                    or attempt_count
                    >= self.clob_identity_revalidation_max_attempts
                ):
                    raise RealCorpusPublicProviderError(
                        "Cached Gamma identity could not be revalidated through CLOB.",
                        reason_codes=tuple(
                            dict.fromkeys(
                                (
                                    *retry_reason_codes,
                                    "gamma_market_identity_cache_clob_revalidation_failed",
                                )
                            )
                        ),
                    ) from exc
                if self.clob_identity_revalidation_retry_seconds:
                    time.sleep(
                        self.clob_identity_revalidation_retry_seconds
                    )
        if not isinstance(payload, dict):
            raise RealCorpusPublicProviderError(
                "CLOB market identity payload is invalid.",
                reason_codes=(
                    "gamma_market_identity_cache_clob_revalidation_failed",
                    "gamma_market_identity_cache_invalid_clob_market_payload",
                ),
            )
        payload_condition_id = str(
            payload.get("condition_id")
            or payload.get("conditionId")
            or payload.get("market")
            or ""
        )
        if payload_condition_id and payload_condition_id != condition_id:
            raise RealCorpusPublicProviderError(
                "Cached Gamma condition id disagrees with CLOB.",
                reason_codes=(
                    "gamma_market_identity_cache_clob_condition_mismatch",
                ),
            )
        payload_slug = str(
            payload.get("market_slug") or payload.get("slug") or ""
        )
        if payload_slug and payload_slug != market["slug"]:
            raise RealCorpusPublicProviderError(
                "Cached Gamma slug disagrees with CLOB.",
                reason_codes=("gamma_market_identity_cache_clob_slug_mismatch",),
            )
        token_by_outcome = _clob_token_by_up_down_outcome(payload.get("tokens"))
        if token_by_outcome is None:
            raise RealCorpusPublicProviderError(
                "CLOB market identity lacks an exact UP/DOWN token mapping.",
                reason_codes=(
                    "gamma_market_identity_cache_clob_tokens_missing",
                ),
            )
        if (
            token_by_outcome["UP"] != market["up_token_id"]
            or token_by_outcome["DOWN"] != market["down_token_id"]
        ):
            raise RealCorpusPublicProviderError(
                "Cached Gamma token ids disagree with CLOB.",
                reason_codes=("gamma_market_identity_cache_clob_token_mismatch",),
            )
        return {
            "passed": True,
            "condition_id": condition_id,
            "slug": market["slug"],
            "up_token_id": market["up_token_id"],
            "down_token_id": market["down_token_id"],
            "clob_market_payload_sha256": canonical_json_sha256(payload),
            "attempt_count": attempt_count,
            "retry_reason_codes": list(dict.fromkeys(retry_reason_codes)),
            "retry_policy_relaxed_identity_checks": False,
        }

    def _fetch_gamma_market_payloads(self, slug: str) -> list[dict[str, Any]]:
        params = urllib.parse.urlencode({"slug": slug})
        payload = self._get_json(f"{self.gamma_markets_endpoint}?{params}")
        if isinstance(payload, list):
            return [dict(row) for row in payload]
        if isinstance(payload, dict):
            data = payload.get("data", payload.get("markets", []))
            if isinstance(data, list):
                return [dict(row) for row in data]
        raise RealCorpusPublicProviderError(
            "Invalid Gamma markets payload.",
            reason_codes=("invalid_gamma_market_payload",),
        )

    def _fetch_clob_book(self, token_id: str) -> dict[str, Any]:
        params = urllib.parse.urlencode({"token_id": token_id})
        payload = self._get_json(f"{self.clob_book_endpoint}?{params}")
        if not isinstance(payload, dict):
            raise RealCorpusPublicProviderError(
                "Invalid CLOB orderbook payload.",
                reason_codes=("invalid_polymarket_orderbook_payload",),
            )
        return dict(payload)

    def _fetch_clob_books(self, token_ids: tuple[str, ...]) -> dict[str, dict[str, Any]]:
        if not token_ids:
            return {}
        if self._fetch_json is not None:
            return {}
        request = urllib.request.Request(
            self.clob_books_endpoint,
            data=json.dumps([{"token_id": token_id} for token_id in token_ids]).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "bigan-v8-polymarket-real-corpus-readonly/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.http_timeout_seconds,
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            return {}
        if not isinstance(payload, list):
            return {}
        rows = [dict(row) for row in payload if isinstance(row, dict)]
        return {str(row.get("asset_id") or ""): row for row in rows if row.get("asset_id")}

    def _normalize_gamma_market_payload(
        self,
        payload: dict[str, Any],
        config: PolymarketRealCorpusRecorderConfig,
        *,
        allow_future_market: bool = False,
    ) -> dict[str, Any] | None:
        slug = str(payload.get("slug") or payload.get("market_slug") or "")
        match = BTC_UPDOWN_SLUG_PATTERN.match(slug)
        if not match:
            return None
        family = BTC_UPDOWN_FAMILY_BY_SLUG[match.group(1)]
        if family not in config.market_families:
            return None
        start_ts = int(match.group(2)) * 1000
        horizon_ms = BTC_UPDOWN_MARKET_HORIZONS_MS[family]
        end_ts = start_ts + horizon_ms
        now_ms = self._current_time_ms()
        if (
            not allow_future_market
            and not self.market_slugs
            and not (start_ts <= now_ms < end_ts)
        ):
            return None
        outcomes = _json_list(payload.get("outcomes"))
        token_ids = _json_list(payload.get("clobTokenIds"))
        token_by_outcome = _token_by_up_down_outcome(outcomes=outcomes, token_ids=token_ids)
        if token_by_outcome is None:
            return None
        condition_id = str(payload.get("conditionId") or payload.get("condition_id") or "")
        if not condition_id:
            return None
        reference_source = str(payload.get("resolutionSource") or "").strip()
        reference_price_start = _reference_price_start_from_payload(payload)
        row = {
            "market_id": condition_id,
            "condition_id": condition_id,
            "slug": slug,
            "market_family": family,
            "horizon_ms": horizon_ms,
            "market_start_ts": start_ts,
            "market_end_ts": end_ts,
            "settlement_ts": _settlement_ts(payload, default=end_ts + 60_000),
            "up_token_id": token_by_outcome["UP"],
            "down_token_id": token_by_outcome["DOWN"],
            "reference_price_source": reference_source,
            "settlement_rule": str(payload.get("description") or payload.get("question") or ""),
            "raw_market_sha256": canonical_json_sha256(payload),
            "raw_public_payload": payload,
            **safety_fields(),
        }
        if reference_price_start is not None:
            row["reference_price_start"] = reference_price_start
            row["reference_price_at_start"] = reference_price_start
            row["reference_price_start_source_type"] = "gamma_market_payload"
        return row

    def _normalize_book_payload(
        self,
        *,
        market: dict[str, Any],
        outcome: str,
        token_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        bids = _price_levels(payload.get("bids"))
        asks = _price_levels(payload.get("asks") if "asks" in payload else payload.get("ask"))
        if not bids or not asks:
            return None
        best_bid_price, best_bid_size = max(bids, key=lambda level: level[0])
        best_ask_price, best_ask_size = min(asks, key=lambda level: level[0])
        timestamp = int(payload.get("timestamp") or self._current_time_ms())
        receive_time = int(payload.get("receive_time") or timestamp)
        available_at_ts = max(timestamp, receive_time)
        return {
            "market_id": market["market_id"],
            "token_id": token_id,
            "outcome": outcome,
            "ts": timestamp,
            "available_at_ts": available_at_ts,
            "bid_price": best_bid_price,
            "ask_price": best_ask_price,
            "mid_price": round((best_bid_price + best_ask_price) / 2.0, 8),
            "bid_size": best_bid_size,
            "ask_size": best_ask_size,
            "liquidity_depth": round(sum(size for _, size in bids + asks), 8),
            **safety_fields(),
        }

    def _normalize_trade_payload(
        self,
        *,
        market: dict[str, Any],
        payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        token_id = str(payload.get("asset") or payload.get("token_id") or "")
        outcome = _expected_outcome_for_token(market=market, token_id=token_id)
        if outcome is None:
            return None
        price = _optional_float(payload.get("price"))
        size = _optional_float(payload.get("size"))
        if price is None or size is None:
            return None
        timestamp = int(float(payload.get("timestamp") or 0) * 1000)
        if timestamp <= 0:
            return None
        return {
            "market_id": market["market_id"],
            "token_id": token_id,
            "outcome": outcome,
            "ts": timestamp,
            "available_at_ts": timestamp,
            "price": price,
            "size": size,
            "side": str(payload.get("side") or "").upper(),
            **safety_fields(),
        }

    def _normalize_binance_kline(
        self,
        row: Any,
        timeframe_ms: int,
        collection_now_ms: int,
    ) -> dict[str, Any] | None:
        if not isinstance(row, list | tuple) or len(row) < 6:
            return None
        ts = int(row[0])
        close_time = ts + timeframe_ms
        if close_time > collection_now_ms:
            return None
        return {
            "ts": ts,
            "close_time": close_time,
            "available_at_ts": close_time,
            "open_price": float(row[1]),
            "high_price": float(row[2]),
            "low_price": float(row[3]),
            "close_price": float(row[4]),
            "volume": float(row[5]),
            "timeframe_ms": timeframe_ms,
            "source": BTC_FEATURE_SOURCE_BINANCE,
        }

    def _normalize_coinbase_candle(
        self,
        row: Any,
        timeframe_ms: int,
        collection_now_ms: int,
    ) -> dict[str, Any] | None:
        if not isinstance(row, list | tuple) or len(row) < 6:
            return None
        ts = int(row[0]) * 1000
        close_time = ts + timeframe_ms
        if close_time > collection_now_ms:
            return None
        return {
            "ts": ts,
            "close_time": close_time,
            "available_at_ts": close_time,
            "open_price": float(row[3]),
            "high_price": float(row[2]),
            "low_price": float(row[1]),
            "close_price": float(row[4]),
            "volume": float(row[5]),
            "timeframe_ms": timeframe_ms,
            "source": BTC_FEATURE_SOURCE_COINBASE,
        }

    def _normalize_kraken_ohlc(
        self,
        row: Any,
        timeframe_ms: int,
        collection_now_ms: int,
    ) -> dict[str, Any] | None:
        if not isinstance(row, list | tuple) or len(row) < 7:
            return None
        ts = int(float(row[0])) * 1000
        close_time = ts + timeframe_ms
        if close_time > collection_now_ms:
            return None
        return {
            "ts": ts,
            "close_time": close_time,
            "available_at_ts": close_time,
            "open_price": float(row[1]),
            "high_price": float(row[2]),
            "low_price": float(row[3]),
            "close_price": float(row[4]),
            "volume": float(row[6]),
            "timeframe_ms": timeframe_ms,
            "source": BTC_FEATURE_SOURCE_KRAKEN,
        }

    def _get_json(self, url: str) -> Any:
        if self._fetch_json is not None:
            return self._fetch_json(url)
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "bigan-v8-polymarket-real-corpus-readonly/1.0"},
            method="GET",
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request, timeout=self.http_timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except TimeoutError as exc:
            raise RealCorpusPublicProviderError(
                "Read-only public HTTP request timed out.",
                reason_codes=("read_only_public_http_timeout",),
            ) from exc
        except urllib.error.HTTPError as exc:
            status = int(getattr(exc, "code", 0) or 0)
            reason_code = (
                "read_only_public_http_server_error"
                if status >= 500
                else "read_only_public_http_error"
            )
            raise RealCorpusPublicProviderError(
                f"Read-only public HTTP request failed with status {status}.",
                reason_codes=(reason_code,),
            ) from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", None)
            if isinstance(reason, TimeoutError) or "timed out" in str(exc).lower():
                raise RealCorpusPublicProviderError(
                    "Read-only public HTTP request timed out.",
                    reason_codes=("read_only_public_http_timeout",),
                ) from exc
            raise RealCorpusPublicProviderError(
                "Read-only public HTTP transport failed.",
                reason_codes=("read_only_public_http_transport_error",),
            ) from exc

    def _current_time_ms(self) -> int:
        return self.current_time_ms if self.current_time_ms is not None else int(time.time() * 1000)


class PolymarketCLOBWebSocketOrderBookSource:
    """Short-lived read-only collector for CLOB market-channel book snapshots."""

    read_only = True
    write_capable = False
    paper_only = True
    capital_at_risk = False
    broker_exchange_write_enabled = False
    live_exchange_write_enabled = False
    polymarket_write_enabled = False
    wallet_signing_enabled = False

    _KEEPALIVE_TOKENS = (b"PONG", b"PING", b"pong", b"ping")

    def __init__(
        self,
        *,
        ws_url: str = DEFAULT_POLYMARKET_CLOB_WS_MARKET_URL,
        timeout_seconds: float = 15.0,
        snapshot_interval_seconds: float = 1.0,
        initial_complete_book_timeout_seconds: float = 15.0,
        custom_feature_enabled: bool = True,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if snapshot_interval_seconds <= 0:
            raise ValueError("snapshot_interval_seconds must be positive")
        if initial_complete_book_timeout_seconds <= 0:
            raise ValueError(
                "initial_complete_book_timeout_seconds must be positive"
            )
        if not ws_url.strip():
            raise ValueError("ws_url is required")
        self.ws_url = ws_url
        self.timeout_seconds = timeout_seconds
        self.snapshot_interval_seconds = snapshot_interval_seconds
        self.initial_complete_book_timeout_seconds = (
            initial_complete_book_timeout_seconds
        )
        self.custom_feature_enabled = custom_feature_enabled
        self._last_market_resolution_payloads: dict[str, dict[str, Any]] = {}

    def book_payloads(self, token_ids: tuple[str, ...]) -> dict[str, dict[str, Any]]:
        token_ids = tuple(dict.fromkeys(str(token_id) for token_id in token_ids if str(token_id)))
        if not token_ids:
            return {}
        try:
            return asyncio.run(self._collect_book_payloads(token_ids))
        except RealCorpusPublicProviderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RealCorpusPublicProviderError(
                f"CLOB websocket orderbook collection failed: {exc}",
                reason_codes=("polymarket_clob_ws_orderbook_collection_failed",),
            ) from exc

    def book_payload_snapshots(
        self,
        token_ids: tuple[str, ...],
    ) -> list[dict[str, dict[str, Any]]]:
        token_ids = tuple(dict.fromkeys(str(token_id) for token_id in token_ids if str(token_id)))
        if not token_ids:
            return []
        try:
            return asyncio.run(self._collect_book_payload_snapshots(token_ids))
        except RealCorpusPublicProviderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RealCorpusPublicProviderError(
                f"CLOB websocket orderbook snapshot collection failed: {exc}",
                reason_codes=("polymarket_clob_ws_orderbook_collection_failed",),
            ) from exc

    def market_resolution_payloads(
        self,
        token_ids: tuple[str, ...],
    ) -> dict[str, dict[str, Any]]:
        token_ids = tuple(dict.fromkeys(str(token_id) for token_id in token_ids if str(token_id)))
        if not token_ids:
            return {}
        return dict(self._last_market_resolution_payloads)

    async def _collect_book_payloads(
        self,
        token_ids: tuple[str, ...],
    ) -> dict[str, dict[str, Any]]:
        target_tokens = set(token_ids)
        book_payloads: dict[str, dict[str, Any]] = {}
        fallback_payloads: dict[str, dict[str, Any]] = {}
        resolution_payloads: dict[str, dict[str, Any]] = {}
        deadline = time.monotonic() + self.timeout_seconds
        connection_error: Exception | None = None
        while time.monotonic() < deadline and set(book_payloads) != target_tokens:
            try:
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=None,
                    ping_timeout=None,
                    close_timeout=5,
                    max_size=2**24,
                ) as ws:
                    await ws.send(
                        orjson.dumps(
                            {
                                "assets_ids": sorted(target_tokens),
                                "type": "market",
                                "custom_feature_enabled": self.custom_feature_enabled,
                            }
                        )
                    )
                    while time.monotonic() < deadline and set(book_payloads) != target_tokens:
                        timeout = max(0.001, deadline - time.monotonic())
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                        except TimeoutError:
                            break
                        receive_time_ms = int(time.time() * 1000)
                        for payload in _decode_market_ws_payloads(raw):
                            self._update_payload_maps(
                                payload=payload,
                                receive_time_ms=receive_time_ms,
                                target_tokens=target_tokens,
                                book_payloads=book_payloads,
                                fallback_payloads=fallback_payloads,
                                resolution_payloads=resolution_payloads,
                            )
            except Exception as exc:  # noqa: BLE001
                connection_error = exc
                await _sleep_until_reconnect(deadline)

        merged = dict(fallback_payloads)
        merged.update(book_payloads)
        if not merged:
            if connection_error is not None:
                raise RealCorpusPublicProviderError(
                    f"CLOB websocket orderbook collection failed: {connection_error}",
                    reason_codes=("polymarket_clob_ws_orderbook_collection_failed",),
                ) from connection_error
            raise RealCorpusPublicProviderError(
                "CLOB websocket emitted no orderbook payloads before timeout.",
                reason_codes=("polymarket_clob_ws_no_orderbooks",),
            )
        self._last_market_resolution_payloads = dict(resolution_payloads)
        return {token_id: merged[token_id] for token_id in token_ids if token_id in merged}

    async def _collect_book_payload_snapshots(
        self,
        token_ids: tuple[str, ...],
    ) -> list[dict[str, dict[str, Any]]]:
        snapshots, resolution_payloads = await self._collect_book_payload_snapshots_and_resolutions(
            token_ids
        )
        self._last_market_resolution_payloads = dict(resolution_payloads)
        return snapshots

    async def _collect_book_payload_snapshots_and_resolutions(
        self,
        token_ids: tuple[str, ...],
    ) -> tuple[list[dict[str, dict[str, Any]]], dict[str, dict[str, Any]]]:
        target_tokens = set(token_ids)
        book_payloads: dict[str, dict[str, Any]] = {}
        fallback_payloads: dict[str, dict[str, Any]] = {}
        resolution_payloads: dict[str, dict[str, Any]] = {}
        snapshots: list[dict[str, dict[str, Any]]] = []
        started_at = time.monotonic()
        deadline = started_at + self.timeout_seconds
        initial_complete_book_deadline = min(
            deadline,
            started_at + self.initial_complete_book_timeout_seconds,
        )
        next_snapshot_at = started_at
        connection_error: Exception | None = None
        complete_book_observed = False
        while time.monotonic() < deadline:
            active_deadline = (
                deadline
                if complete_book_observed
                else initial_complete_book_deadline
            )
            if time.monotonic() >= active_deadline:
                break
            try:
                open_timeout = max(
                    0.001,
                    min(10.0, active_deadline - time.monotonic()),
                )
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=None,
                    ping_timeout=None,
                    close_timeout=5,
                    max_size=2**24,
                    open_timeout=open_timeout,
                ) as ws:
                    await ws.send(
                        orjson.dumps(
                            {
                                "assets_ids": sorted(target_tokens),
                                "type": "market",
                                "custom_feature_enabled": self.custom_feature_enabled,
                            }
                        )
                    )
                    while time.monotonic() < deadline:
                        active_deadline = (
                            deadline
                            if complete_book_observed
                            else initial_complete_book_deadline
                        )
                        if time.monotonic() >= active_deadline:
                            break
                        now = time.monotonic()
                        timeout = max(
                            0.001,
                            min(active_deadline, next_snapshot_at) - now,
                        )
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                        except TimeoutError:
                            raw = None
                        if raw is not None:
                            receive_time_ms = int(time.time() * 1000)
                            for payload in _decode_market_ws_payloads(raw):
                                self._update_payload_maps(
                                    payload=payload,
                                    receive_time_ms=receive_time_ms,
                                    target_tokens=target_tokens,
                                    book_payloads=book_payloads,
                                    fallback_payloads=fallback_payloads,
                                    resolution_payloads=resolution_payloads,
                                )
                        if time.monotonic() >= next_snapshot_at:
                            merged = dict(fallback_payloads)
                            merged.update(book_payloads)
                            if merged:
                                if target_tokens.issubset(merged):
                                    complete_book_observed = True
                                observation_time_ms = int(time.time() * 1000)
                                snapshots.append(
                                    _observed_snapshot_payloads(
                                        merged=merged,
                                        token_ids=token_ids,
                                        observation_time_ms=observation_time_ms,
                                    )
                                )
                            next_snapshot_at = (
                                time.monotonic() + self.snapshot_interval_seconds
                            )
            except Exception as exc:  # noqa: BLE001
                connection_error = exc
                reconnect_deadline = (
                    deadline
                    if complete_book_observed
                    else initial_complete_book_deadline
                )
                await _sleep_until_reconnect(reconnect_deadline)

        if not snapshots:
            if connection_error is not None:
                raise RealCorpusPublicProviderError(
                    "CLOB websocket did not establish a complete initial orderbook "
                    f"before the bounded timeout: {connection_error}",
                    reason_codes=(
                        "polymarket_clob_ws_initial_complete_book_timeout",
                        "polymarket_clob_ws_orderbook_collection_failed",
                    ),
                ) from connection_error
            raise RealCorpusPublicProviderError(
                "CLOB websocket emitted no complete initial orderbook before timeout.",
                reason_codes=(
                    "polymarket_clob_ws_initial_complete_book_timeout",
                    "polymarket_clob_ws_no_orderbooks",
                ),
            )
        return snapshots, resolution_payloads

    def _update_payload_maps(
        self,
        *,
        payload: dict[str, Any],
        receive_time_ms: int,
        target_tokens: set[str],
        book_payloads: dict[str, dict[str, Any]],
        fallback_payloads: dict[str, dict[str, Any]],
        resolution_payloads: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        normalized_payload = _market_ws_payload_for_parse(
            payload=payload,
            receive_time_ms=receive_time_ms,
        )
        if normalized_payload is None:
            return
        try:
            event = parse_event(normalized_payload, receive_time_ms=receive_time_ms)
        except UnknownEvent:
            return
        except Exception:
            return
        if isinstance(event, BookEvent) and event.asset_id in target_tokens:
            book_payloads[event.asset_id] = _book_event_payload(event)
            return
        if isinstance(event, BestBidAskEvent) and event.asset_id in target_tokens:
            top_payload = _top_of_book_payload(
                asset_id=event.asset_id,
                market=event.market,
                timestamp=int(event.timestamp),
                best_bid=event.best_bid,
                best_ask=event.best_ask,
                receive_time=event.receive_time,
                source_event_type="best_bid_ask",
            )
            if top_payload is not None:
                fallback_payloads[event.asset_id] = top_payload
            return
        if isinstance(event, PriceChangeEvent):
            for change in event.price_changes:
                if change.asset_id not in target_tokens:
                    continue
                top_payload = _top_of_book_payload(
                    asset_id=change.asset_id,
                    market=event.market,
                    timestamp=int(event.timestamp),
                    best_bid=change.best_bid,
                    best_ask=change.best_ask,
                    receive_time=event.receive_time,
                    source_event_type="price_change",
                )
                if top_payload is not None:
                    fallback_payloads[change.asset_id] = top_payload
            return
        if isinstance(event, MarketResolvedEvent) and resolution_payloads is not None:
            resolution_payload = _market_resolved_event_payload(event)
            if resolution_payload is None:
                return
            event_tokens = set(resolution_payload.get("assets_ids") or ())
            winning_asset_id = str(resolution_payload.get("winning_asset_id") or "")
            if winning_asset_id in target_tokens or event_tokens & target_tokens:
                resolution_payloads[str(resolution_payload["market"])] = resolution_payload


async def _sleep_until_reconnect(deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining > 0:
        await asyncio.sleep(min(1.0, remaining))


def _token_ids_for_markets(markets: list[dict[str, Any]]) -> tuple[str, ...]:
    token_ids: list[str] = []
    for market in markets:
        token_ids.extend([str(market["up_token_id"]), str(market["down_token_id"])])
    return tuple(dict.fromkeys(token_ids))


def _with_orderbook_collection_end(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    collection_end_ts = max(int(row.get("available_at_ts") or row["ts"]) for row in rows)
    return [dict(row, collection_end_ts=collection_end_ts) for row in rows]


def _annotate_orderbook_rows(
    rows: list[dict[str, Any]],
    *,
    source_type: str,
    rest_fallback: bool,
    fallback_reason_codes: tuple[str, ...],
) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "orderbook_source_type": source_type,
            "orderbook_rest_fallback_used": rest_fallback,
            "orderbook_fallback_reason_codes": list(fallback_reason_codes),
        }
        for row in rows
    ]


def _annotate_market_identity(
    row: dict[str, Any],
    *,
    source_type: str,
    fetched_at_ts: int,
    cache_fallback: bool,
    fallback_reason_codes: tuple[str, ...],
    cache_entry_sha256: str | None,
    cache_age_ms: int | None,
    clob_validation: dict[str, Any] | None,
) -> dict[str, Any]:
    annotated = {
        **row,
        "market_identity_source_type": source_type,
        "market_identity_fetched_at_ts": fetched_at_ts,
        "market_identity_cache_fallback_used": cache_fallback,
        "market_identity_cache_fallback_reason_codes": list(
            fallback_reason_codes
        ),
        "market_identity_cache_entry_sha256": cache_entry_sha256,
        "market_identity_cache_age_ms": cache_age_ms,
        "market_identity_cache_provenance_valid": True,
        "market_identity_clob_revalidation_required": cache_fallback,
        "market_identity_clob_revalidation_passed": (
            bool(clob_validation and clob_validation.get("passed"))
            if cache_fallback
            else None
        ),
        "market_identity_clob_revalidation": clob_validation,
        "market_identity_live_orderbook_validation_required": True,
    }
    return annotated


def _gamma_cache_fallback_allowed(reason_codes: tuple[str, ...]) -> bool:
    allowed = {
        "read_only_public_http_timeout",
        "read_only_public_http_transport_error",
        "read_only_public_http_server_error",
    }
    return bool(set(reason_codes) & allowed)


def _decode_market_ws_payloads(raw: bytes | str) -> list[dict[str, Any]]:
    raw_bytes = raw if isinstance(raw, bytes) else raw.encode("utf-8")
    stripped = raw_bytes.strip()
    if stripped in PolymarketCLOBWebSocketOrderBookSource._KEEPALIVE_TOKENS:
        return []
    try:
        payload = orjson.loads(raw_bytes)
    except orjson.JSONDecodeError:
        return []
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _market_ws_payload_for_parse(
    *,
    payload: dict[str, Any],
    receive_time_ms: int,
) -> dict[str, Any] | None:
    normalized = dict(payload)
    if "event_type" not in normalized:
        inferred = _infer_market_ws_event_type(normalized)
        if inferred is None:
            return None
        normalized["event_type"] = inferred
    if "timestamp" not in normalized:
        normalized["timestamp"] = str(receive_time_ms)
    if "asks" not in normalized and "ask" in normalized:
        normalized["asks"] = normalized["ask"]
    return normalized


def _infer_market_ws_event_type(payload: dict[str, Any]) -> str | None:
    if "winning_asset_id" in payload or "winning_outcome" in payload:
        return "market_resolved"
    if "price_changes" in payload:
        return "price_change"
    if "asset_id" in payload and "bids" in payload and ("asks" in payload or "ask" in payload):
        return "book"
    if "asset_id" in payload and ("best_bid" in payload or "best_ask" in payload):
        return "best_bid_ask"
    return None


def _book_event_payload(event: BookEvent) -> dict[str, Any]:
    return {
        "market": event.market,
        "asset_id": event.asset_id,
        "timestamp": int(event.timestamp),
        "receive_time": event.receive_time,
        "source_channel": POLYMARKET_CLOB_WS_SOURCE_CHANNEL,
        "source_event_type": "book",
        "hash": event.hash,
        "bids": _price_level_dicts(event.bids),
        "asks": _price_level_dicts(event.asks),
    }


def _price_level_dicts(levels: list[Any]) -> list[dict[str, str]]:
    return [{"price": str(level.price), "size": str(level.size)} for level in levels]


def _top_of_book_payload(
    *,
    asset_id: str,
    market: str,
    timestamp: int,
    best_bid: Any,
    best_ask: Any,
    receive_time: int | None,
    source_event_type: str,
) -> dict[str, Any] | None:
    bid = _optional_float(best_bid)
    ask = _optional_float(best_ask)
    if bid is None or ask is None:
        return None
    return {
        "market": market,
        "asset_id": asset_id,
        "timestamp": timestamp,
        "receive_time": receive_time,
        "source_channel": POLYMARKET_CLOB_WS_SOURCE_CHANNEL,
        "source_event_type": source_event_type,
        "bids": [{"price": str(bid), "size": "0"}],
        "asks": [{"price": str(ask), "size": "0"}],
    }


def _observed_snapshot_payloads(
    *,
    merged: dict[str, dict[str, Any]],
    token_ids: tuple[str, ...],
    observation_time_ms: int,
) -> dict[str, dict[str, Any]]:
    observed: dict[str, dict[str, Any]] = {}
    for token_id in token_ids:
        payload = merged.get(token_id)
        if payload is None:
            continue
        snapshot = dict(payload)
        snapshot["source_book_timestamp"] = snapshot.get("timestamp")
        snapshot["timestamp"] = observation_time_ms
        snapshot["receive_time"] = observation_time_ms
        snapshot["source_event_type"] = str(snapshot.get("source_event_type") or "observed_book")
        observed[token_id] = snapshot
    return observed


def _market_resolved_event_payload(event: MarketResolvedEvent) -> dict[str, Any] | None:
    market = str(getattr(event, "market", "") or "")
    winning_asset_id = str(getattr(event, "winning_asset_id", "") or "")
    winning_outcome = str(getattr(event, "winning_outcome", "") or "")
    if not market or not winning_asset_id or not winning_outcome:
        return None
    assets_ids = getattr(event, "assets_ids", None)
    outcomes = getattr(event, "outcomes", None)
    return {
        "event_type": "market_resolved",
        "market": market,
        "timestamp": int(event.timestamp),
        "receive_time": event.receive_time,
        "assets_ids": [str(asset_id) for asset_id in assets_ids]
        if isinstance(assets_ids, list)
        else [],
        "outcomes": [str(outcome) for outcome in outcomes] if isinstance(outcomes, list) else [],
        "winning_asset_id": winning_asset_id,
        "winning_outcome": winning_outcome,
    }


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _token_by_up_down_outcome(
    *,
    outcomes: list[Any],
    token_ids: list[Any],
) -> dict[str, str] | None:
    if len(outcomes) != len(token_ids):
        return None
    token_by_outcome: dict[str, str] = {}
    for outcome, token_id in zip(outcomes, token_ids, strict=True):
        normalized = str(outcome).strip().upper()
        if normalized == "UP":
            token_by_outcome["UP"] = str(token_id)
        elif normalized == "DOWN":
            token_by_outcome["DOWN"] = str(token_id)
    return token_by_outcome if set(token_by_outcome) == {"UP", "DOWN"} else None


def _clob_token_by_up_down_outcome(value: Any) -> dict[str, str] | None:
    if not isinstance(value, list):
        return None
    outcomes: list[str] = []
    token_ids: list[str] = []
    for token in value:
        if not isinstance(token, dict):
            continue
        outcome = str(
            token.get("outcome")
            or token.get("name")
            or token.get("label")
            or ""
        )
        token_id = str(
            token.get("token_id")
            or token.get("tokenId")
            or token.get("asset_id")
            or ""
        )
        if outcome and token_id:
            outcomes.append(outcome)
            token_ids.append(token_id)
    return _token_by_up_down_outcome(outcomes=outcomes, token_ids=token_ids)


def _settlement_ts(payload: dict[str, Any], *, default: int) -> int:
    for key in ("closedTime", "closed_time", "endDate", "end_date_iso"):
        value = payload.get(key)
        if value:
            parsed = _parse_iso_millis(str(value))
            if parsed is not None:
                return max(parsed, default)
    return default


def _parse_iso_millis(value: str) -> int | None:
    try:
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)
    except ValueError:
        return None


def _price_levels(value: Any) -> list[tuple[float, float]]:
    if not isinstance(value, list):
        return []
    levels = []
    for row in value:
        if not isinstance(row, dict):
            continue
        price = _optional_float(row.get("price"))
        size = _optional_float(row.get("size"))
        if price is not None and size is not None:
            levels.append((price, size))
    return levels


def _reference_price_fields_for_resolution(
    *,
    market: dict[str, Any],
    event_payload: dict[str, Any],
    gamma_market_payload: dict[str, Any],
) -> dict[str, Any]:
    raw_market_payload = market.get("raw_public_payload")
    for source_type, payload in (
        ("raw_market_payload", raw_market_payload),
        ("gamma_event_metadata", event_payload),
        ("gamma_market_payload", gamma_market_payload),
    ):
        if not isinstance(payload, dict):
            continue
        pair = _reference_price_pair_from_payload(payload)
        if pair is None:
            continue
        start, end = pair
        return {
            "reference_price_start": start,
            "reference_price_end": end,
            "reference_price_source_type": source_type,
        }
    return {}


def _reference_price_start_from_payload(payload: dict[str, Any]) -> float | None:
    for candidate in _reference_price_candidates(payload):
        start = _first_positive_float(
            candidate,
            "priceToBeat",
            "price_to_beat",
            "referencePriceStart",
            "reference_price_start",
            "reference_price_at_start",
        )
        if start is not None:
            return start
    return None


def _reference_price_pair_from_payload(payload: dict[str, Any]) -> tuple[float, float] | None:
    for candidate in _reference_price_candidates(payload):
        start = _first_positive_float(
            candidate,
            "priceToBeat",
            "price_to_beat",
            "referencePriceStart",
            "reference_price_start",
            "reference_price_at_start",
            "openPrice",
            "open_price",
            "start_price",
        )
        end = _first_positive_float(
            candidate,
            "referencePriceEnd",
            "reference_price_end",
            "reference_price_at_end",
            "finalPrice",
            "final_price",
            "closePrice",
            "close_price",
            "target_price",
        )
        if start is not None and end is not None:
            return start, end
    return None


def _reference_price_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    metadata = payload.get("eventMetadata")
    if isinstance(metadata, dict):
        candidates.append(metadata)
    candidates.append(payload)
    markets = payload.get("markets")
    if isinstance(markets, list):
        candidates.extend(dict(row) for row in markets if isinstance(row, dict))
    return candidates


def _first_positive_float(payload: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _optional_float(payload.get(key))
        if value is not None and value > 0.0:
            return value
    return None


def _optional_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric


def _expected_outcome_for_token(*, market: dict[str, Any], token_id: str) -> str | None:
    if token_id == str(market.get("up_token_id")):
        return "UP"
    if token_id == str(market.get("down_token_id")):
        return "DOWN"
    return None


def _iso_millis(ts_ms: int) -> str:
    return (
        datetime.fromtimestamp(ts_ms / 1000, tz=UTC)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _kraken_ohlc_rows(payload: dict[str, Any]) -> list[Any]:
    result = payload.get("result")
    if not isinstance(result, dict):
        return []
    for key, value in result.items():
        if key == "last":
            continue
        if isinstance(value, list):
            return value
    return []


def _kraken_interval(timeframe_ms: int) -> int:
    mapping = {
        60_000: 1,
        300_000: 5,
        900_000: 15,
        3_600_000: 60,
    }
    if timeframe_ms not in mapping:
        raise RealCorpusPublicProviderError(
            f"Unsupported BTC feature candle timeframe_ms={timeframe_ms}",
            reason_codes=("unsupported_btc_feature_candle_timeframe",),
        )
    return mapping[timeframe_ms]


def _binance_interval(timeframe_ms: int) -> str:
    mapping = {
        60_000: "1m",
        300_000: "5m",
        900_000: "15m",
        3_600_000: "1h",
    }
    if timeframe_ms not in mapping:
        raise RealCorpusPublicProviderError(
            f"Unsupported BTC feature candle timeframe_ms={timeframe_ms}",
            reason_codes=("unsupported_btc_feature_candle_timeframe",),
        )
    return mapping[timeframe_ms]


def _resolution_status_from_payload(payload: dict[str, Any]) -> str:
    prices = _json_list(payload.get("outcomePrices"))
    if len(prices) == 2 and all(str(price) == "0.5" for price in prices):
        return "unknown_50_50"
    return "normal"


def _payout_resolution_from_gamma_payload(
    *,
    payload: dict[str, Any],
    market: dict[str, Any],
    current_time_ms: int,
) -> dict[str, Any] | None:
    if current_time_ms < int(market["market_end_ts"]):
        return None
    outcomes = _json_list(payload.get("outcomes"))
    prices = [_optional_float(value) for value in _json_list(payload.get("outcomePrices"))]
    if len(outcomes) != 2 or len(prices) != 2 or any(price is None for price in prices):
        return None
    payout_by_outcome = {
        str(outcome).strip().upper(): float(price)
        for outcome, price in zip(outcomes, prices, strict=True)
        if str(outcome).strip().upper() in {"UP", "DOWN"} and price is not None
    }
    if set(payout_by_outcome) != {"UP", "DOWN"}:
        return None
    up = payout_by_outcome["UP"]
    down = payout_by_outcome["DOWN"]
    if _near_payout(up, 1.0) and _near_payout(down, 0.0):
        return {
            "resolution_status": "normal",
            "resolved_outcome": "UP",
            "payout_up": 1.0,
            "payout_down": 0.0,
        }
    if _near_payout(up, 0.0) and _near_payout(down, 1.0):
        return {
            "resolution_status": "normal",
            "resolved_outcome": "DOWN",
            "payout_up": 0.0,
            "payout_down": 1.0,
        }
    if _near_payout(up, 0.5) and _near_payout(down, 0.5):
        return {
            "resolution_status": "unknown_50_50",
            "resolved_outcome": "UNKNOWN_50_50",
            "payout_up": 0.5,
            "payout_down": 0.5,
        }
    return None


def _payout_resolution_from_market_resolved_payload(
    *,
    payload: dict[str, Any] | None,
    market: dict[str, Any],
) -> dict[str, Any] | None:
    if payload is None:
        return None
    winning_asset_id = str(payload.get("winning_asset_id") or "")
    winning_outcome = str(payload.get("winning_outcome") or "").strip().upper()
    if winning_asset_id == str(market["up_token_id"]) or winning_outcome == "UP":
        return {
            "resolution_status": "normal",
            "resolved_outcome": "UP",
            "payout_up": 1.0,
            "payout_down": 0.0,
        }
    if winning_asset_id == str(market["down_token_id"]) or winning_outcome == "DOWN":
        return {
            "resolution_status": "normal",
            "resolved_outcome": "DOWN",
            "payout_up": 0.0,
            "payout_down": 1.0,
        }
    return None


def _payout_resolution_from_clob_market_payload(
    *,
    payload: dict[str, Any] | None,
    market: dict[str, Any],
) -> dict[str, Any] | None:
    if not payload or payload.get("closed") is not True:
        return None
    tokens = payload.get("tokens")
    if not isinstance(tokens, list):
        return None
    winning_token_ids = {
        str(token.get("token_id") or token.get("asset_id") or "")
        for token in tokens
        if isinstance(token, dict) and token.get("winner") is True
    }
    if str(market["up_token_id"]) in winning_token_ids:
        return {
            "resolution_status": "normal",
            "resolved_outcome": "UP",
            "payout_up": 1.0,
            "payout_down": 0.0,
        }
    if str(market["down_token_id"]) in winning_token_ids:
        return {
            "resolution_status": "normal",
            "resolved_outcome": "DOWN",
            "payout_up": 0.0,
            "payout_down": 1.0,
        }
    prices_by_token = {
        str(token.get("token_id") or token.get("asset_id") or ""): _optional_float(
            token.get("price")
        )
        for token in tokens
        if isinstance(token, dict)
    }
    up_price = prices_by_token.get(str(market["up_token_id"]))
    down_price = prices_by_token.get(str(market["down_token_id"]))
    if up_price is None or down_price is None:
        return None
    if _near_payout(up_price, 1.0) and _near_payout(down_price, 0.0):
        return {
            "resolution_status": "normal",
            "resolved_outcome": "UP",
            "payout_up": 1.0,
            "payout_down": 0.0,
        }
    if _near_payout(up_price, 0.0) and _near_payout(down_price, 1.0):
        return {
            "resolution_status": "normal",
            "resolved_outcome": "DOWN",
            "payout_up": 0.0,
            "payout_down": 1.0,
        }
    return None


def _near_payout(value: float, expected: float) -> bool:
    return abs(value - expected) <= 1e-9
