"""Strict Gamma parsing and deterministic market-selection tests."""

from __future__ import annotations

from dataclasses import replace

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
