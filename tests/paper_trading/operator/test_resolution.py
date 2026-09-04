from __future__ import annotations

from copy import deepcopy

import pytest

from bigan.paper_trading.operator.discovery import parse_gamma_markets
from bigan.paper_trading.operator.resolution import ResolutionError, parse_gamma_resolution


def _row() -> dict[str, object]:
    return {
        "id": "market-1",
        "conditionId": "condition-1",
        # Slug timestamps are seconds, while the explicit fields below are ms.
        "slug": "btc-updown-15m-1",
        "question": "BTC 15 minute window 1000",
        "underlying": "BTC",
        "marketType": "binary_up_down",
        "windowDurationMs": 900_000,
        "start_ts_ms": 1_000,
        "end_ts_ms": 901_000,
        "outcomes": ["Up", "Down"],
        "clobTokenIds": ["yes-token", "no-token"],
        "active": True,
        "closed": False,
        "acceptingOrders": True,
        "resolutionSource": "chainlink-btc-usd",
        "resolutionIdentity": "rule-v1",
        "referencePriceAtStart": 100_000,
    }


def _market():
    return parse_gamma_markets(
        [_row()],
        source_endpoint="https://gamma-api.polymarket.com/markets",
        discovered_at_ms=2_000,
    )[0]


def _resolved() -> dict[str, object]:
    row = _row()
    row.update(
        {
            "closed": True,
            "active": False,
            "acceptingOrders": False,
            "resolved": True,
            "outcomePrices": ["1", "0"],
            "resolutionTimestamp": 902_000,
        }
    )
    return row


def test_final_resolution_has_strict_binary_payout_and_provenance() -> None:
    result = parse_gamma_resolution(
        [_resolved()],
        market=_market(),
        source_endpoint="https://gamma-api.polymarket.com/markets",
        received_at_ms=903_000,
    )
    assert result is not None
    assert result.yes_payout == 1.0
    assert result.window_id == _market().window_id
    assert result.source_reference.startswith("https://gamma-api.polymarket.com/markets#")


def test_non_final_resolution_stays_pending() -> None:
    row = _resolved()
    row["resolved"] = False
    assert (
        parse_gamma_resolution(
            [row],
            market=_market(),
            source_endpoint="https://gamma-api.polymarket.com/markets",
            received_at_ms=903_000,
        )
        is None
    )


@pytest.mark.parametrize(
    "mutation",
    [
        {"conditionId": "other"},
        {"clobTokenIds": ["wrong", "no-token"]},
        {"outcomePrices": ["0.8", "0.2"]},
        {"resolutionSource": "other"},
        {"resolutionIdentity": "other"},
    ],
)
def test_resolution_identity_or_payout_mismatch_fails_closed(
    mutation: dict[str, object],
) -> None:
    row = _resolved()
    row.update(mutation)
    with pytest.raises(ResolutionError):
        parse_gamma_resolution(
            [row],
            market=_market(),
            source_endpoint="https://gamma-api.polymarket.com/markets",
            received_at_ms=903_000,
        )


def test_missing_or_duplicate_market_identity_fails_closed() -> None:
    row = _resolved()
    with pytest.raises(ResolutionError, match="exactly one"):
        parse_gamma_resolution([], market=_market(), source_endpoint="https://x", received_at_ms=1)
    with pytest.raises(ResolutionError, match="exactly one"):
        parse_gamma_resolution(
            [row, deepcopy(row)],
            market=_market(),
            source_endpoint="https://x",
            received_at_ms=1,
        )
