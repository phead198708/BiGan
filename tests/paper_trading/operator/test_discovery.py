"""Strict Gamma parsing and deterministic market-selection tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from bigan.paper_trading.operator.discovery import (
    DiscoveryFilters,
    MarketDiscoveryError,
    parse_gamma_markets,
    select_market_windows,
)

NOW = 2_000_000
DURATION = 900_000
ENDPOINT = "https://gamma-api.polymarket.com/markets"


def _row(
    market_id: str,
    *,
    start: int,
    active: bool = True,
    closed: bool = False,
    accepting_orders: bool = True,
) -> dict[str, object]:
    return {
        "id": market_id,
        "conditionId": f"condition-{market_id}",
        "slug": f"btc-updown-15m-{start // 1000}",
        "question": "Bitcoin Up or Down - 15 minutes",
        "underlying": "BTC",
        "marketType": "binary_up_down",
        "windowDurationMs": DURATION,
        "start_ts_ms": start,
        "end_ts_ms": start + DURATION,
        "active": active,
        "closed": closed,
        "acceptingOrders": accepting_orders,
        "outcomes": '["Up","Down"]',
        "clobTokenIds": f'["yes-{market_id}","no-{market_id}"]',
        "resolutionSource": "polymarket-final-outcome",
        "resolutionIdentity": f"resolution-{market_id}",
    }


def _filters() -> DiscoveryFilters:
    return DiscoveryFilters(
        underlying="BTC",
        market_type="binary_up_down",
        window_duration_ms=DURATION,
        slug_pattern=r"btc-updown-15m-\d+",
        max_preopen_ms=2 * DURATION,
    )


def test_discovery_rejects_unsupported_one_hour_family() -> None:
    with pytest.raises(ValueError, match="only 5m and 15m"):
        DiscoveryFilters(
            underlying="BTC",
            market_type="binary_up_down",
            window_duration_ms=3_600_000,
        )


def test_selects_unique_active_and_deterministic_next_window() -> None:
    rows = [
        _row("next-late", start=NOW + DURATION),
        _row("active", start=NOW - 100_000),
        _row("next-early", start=NOW + 100_000),
    ]
    candidates = parse_gamma_markets(
        rows,
        source_endpoint=ENDPOINT,
        discovered_at_ms=NOW,
    )
    selected = select_market_windows(candidates, filters=_filters(), now_ms=NOW)

    assert selected.current is not None
    assert selected.current.market_id == "active"
    assert selected.next is not None
    assert selected.next.market_id == "next-early"
    assert selected.current.source_endpoint == ENDPOINT
    assert selected.current.discovered_at_ms == NOW
    assert selected.current.yes_token_id == "yes-active"
    assert selected.current.no_token_id == "no-active"


@pytest.mark.parametrize(
    "mutation",
    [
        {"clobTokenIds": '["same","same"]'},
        {"clobTokenIds": '["only-one"]'},
        {"outcomes": '["Up"]'},
        {"conditionId": ""},
        {"end_ts_ms": NOW},
    ],
)
def test_missing_duplicate_or_invalid_market_identity_is_rejected(
    mutation: dict[str, object],
) -> None:
    row = _row("broken", start=NOW)
    row.update(mutation)
    with pytest.raises(MarketDiscoveryError, match="invalid market payload"):
        parse_gamma_markets([row], source_endpoint=ENDPOINT, discovered_at_ms=NOW)


def test_ambiguous_equal_rank_candidates_fail_closed() -> None:
    candidates = parse_gamma_markets(
        [_row("one", start=NOW), _row("two", start=NOW)],
        source_endpoint=ENDPOINT,
        discovered_at_ms=NOW,
    )
    with pytest.raises(MarketDiscoveryError, match="ambiguous") as error:
        select_market_windows(candidates, filters=_filters(), now_ms=NOW)
    assert error.value.reason_code == "ambiguous_current_market"


def test_inactive_closed_and_out_of_range_markets_are_not_selected() -> None:
    candidates = parse_gamma_markets(
        [
            _row("inactive", start=NOW, active=False),
            _row("closed", start=NOW, closed=True),
            _row("not-orderable", start=NOW, accepting_orders=False),
            _row("too-far", start=NOW + 3 * DURATION),
        ],
        source_endpoint=ENDPOINT,
        discovered_at_ms=NOW,
    )
    with pytest.raises(MarketDiscoveryError, match="no eligible") as error:
        select_market_windows(candidates, filters=_filters(), now_ms=NOW)
    assert error.value.reason_code == "no_eligible_market"


def test_filters_do_not_use_fuzzy_title_substrings() -> None:
    candidate = parse_gamma_markets(
        [_row("one", start=NOW)],
        source_endpoint=ENDPOINT,
        discovered_at_ms=NOW,
    )[0]
    mismatched = replace(candidate, underlying="ETH", title="BTC maybe maybe")
    with pytest.raises(MarketDiscoveryError, match="no eligible"):
        select_market_windows((mismatched,), filters=_filters(), now_ms=NOW)


@pytest.mark.parametrize("payload", [{}, {"markets": {}}, "broken", [None]])
def test_malformed_response_fails_closed(payload: object) -> None:
    with pytest.raises(MarketDiscoveryError, match="invalid.*payload"):
        parse_gamma_markets(
            payload,
            source_endpoint=ENDPOINT,
            discovered_at_ms=NOW,
        )


def test_real_gamma_shape_uses_exact_slug_family_not_fuzzy_title() -> None:
    row = _row("real", start=1_000_000)
    row.pop("underlying")
    row.pop("marketType")
    row.pop("windowDurationMs")
    row.pop("resolutionIdentity")
    row.pop("start_ts_ms")
    parsed = parse_gamma_markets(
        [row],
        source_endpoint=ENDPOINT,
        discovered_at_ms=NOW,
    )[0]
    assert parsed.underlying == "BTC"
    assert parsed.market_type == "binary_up_down"
    assert parsed.window_duration_ms == DURATION
    assert parsed.start_ts_ms == 1_000_000
    assert parsed.resolution_identity.startswith("condition:condition-real")
    assert parsed.reference_price_at_start is None


@pytest.mark.parametrize("duration,label", [(300_000, "5m"), (900_000, "15m")])
@pytest.mark.parametrize("with_event_start", [False, True])
def test_gamma_listing_time_is_not_window_start(duration, label, with_event_start):
    # Recorded public Gamma shape: listing time is a day before eventStartTime.
    start = 1_788_500_700_000
    row = _row("4182957", start=start)
    for key in ("start_ts_ms", "end_ts_ms", "windowDurationMs"):
        row.pop(key)
    row.update(slug=f"btc-updown-{label}-{start // 1000}",
               startDate="2026-09-03T05:53:30.078891Z", startDateIso="2026-09-03",
               endDate=datetime.fromtimestamp((start + duration) / 1000, UTC).isoformat())
    if with_event_start:
        row["eventStartTime"] = "2026-09-04T05:45:00Z"
    market = parse_gamma_markets([row], source_endpoint=ENDPOINT, discovered_at_ms=start)[0]
    assert market.start_ts_ms == start
    assert market.end_ts_ms == start + duration
    assert market.window_duration_ms == duration
    assert market.reference_price_at_start is None  # No invented strike fallback.


@pytest.mark.parametrize("mutation", [
    {"eventStartTime": "1970-01-01T00:33:21Z"},
    {"eventStartTime": "invalid"},
    {"start_ts_ms": NOW + 1},
    {"slug": "btc-updown-5m-2000"},
])
def test_conflicting_or_invalid_window_identity_fails_closed(mutation):
    row = _row("conflict", start=NOW)
    row.update(mutation)
    with pytest.raises(MarketDiscoveryError, match="invalid market payload"):
        parse_gamma_markets([row], source_endpoint=ENDPOINT, discovered_at_ms=NOW)


def test_listing_date_alone_is_not_a_window_identity():
    row = _row("unknown", start=NOW)
    row.pop("start_ts_ms")
    row.update(slug="noncanonical", startDate="1970-01-01T00:33:20Z")
    with pytest.raises(MarketDiscoveryError, match="invalid market payload"):
        parse_gamma_markets([row], source_endpoint=ENDPOINT, discovered_at_ms=NOW)
