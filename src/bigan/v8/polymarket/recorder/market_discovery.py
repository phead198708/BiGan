"""Market discovery helpers for Polymarket raw corpus recording."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.corpus import BTC_UPDOWN_MARKET_HORIZONS_MS
from bigan.v8.polymarket.corpus.contracts import safety_fields
from bigan.v8.polymarket.recorder.contracts import (
    PolymarketRealCorpusRecorderConfig,
)

MOCK_RECORDER_BASE_TS = 1_780_400_000_000
DEFAULT_SETTLEMENT_RULE = (
    "UP wins if the official Polymarket BTC reference price at market end is greater "
    "than the official Polymarket BTC reference price at market start; otherwise DOWN wins."
)


def discover_mock_market_rows(config: PolymarketRealCorpusRecorderConfig) -> list[dict[str, Any]]:
    """Return deterministic BTC UP/DOWN market metadata for tests and local smoke runs."""

    rows: list[dict[str, Any]] = []
    for index, family in enumerate(config.market_families):
        horizon_ms = BTC_UPDOWN_MARKET_HORIZONS_MS[family]
        start_ts = MOCK_RECORDER_BASE_TS + index * 7_200_000
        end_ts = start_ts + horizon_ms
        source = (
            ""
            if config.inject_missing_reference_source and index == 0
            else config.official_settlement_reference_source
        )
        market_id = f"real-corpus-{family}-{index}"
        rows.append(
            {
                "market_id": market_id,
                "condition_id": f"0xrealcorpuscondition{index:04d}",
                "slug": f"btc-updown-{_family_slug_suffix(family)}-{start_ts // 1000}",
                "market_family": family,
                "horizon_ms": horizon_ms,
                "market_start_ts": start_ts,
                "market_end_ts": end_ts,
                "settlement_ts": end_ts + 60_000,
                "up_token_id": f"real-corpus-up-token-{index}",
                "down_token_id": f"real-corpus-down-token-{index}",
                "reference_price_source": source,
                "settlement_rule": DEFAULT_SETTLEMENT_RULE,
                "reference_price_start": 65_000.0 + index * 125.0,
                "raw_market_sha256": canonical_json_sha256(
                    {"market_id": market_id, "family": family, "source": "mock_public_data"}
                ),
                "raw_public_payload": {
                    "mock_public_data": True,
                    "family": family,
                    "index": index,
                },
                **safety_fields(),
            }
        )
    return rows


class PolymarketGammaReadOnlyClient:
    """Small read-only Gamma market discovery client.

    The operator uses mocked data in CI. This client is intentionally read-only and
    provides a narrow seam for manual collection runs.
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
        endpoint: str = "https://gamma-api.polymarket.com/markets",
        timeout_seconds: float = 10.0,
    ) -> None:
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds

    def fetch_market_payloads(self, *, query: str = "bitcoin up or down") -> list[dict[str, Any]]:
        params = urllib.parse.urlencode({"q": query, "active": "true"})
        request = urllib.request.Request(
            f"{self.endpoint}?{params}",
            headers={"User-Agent": "bigan-v8-real-corpus-recorder/1.0"},
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if isinstance(payload, list):
            return [dict(row) for row in payload]
        if isinstance(payload, dict):
            data = payload.get("data", payload.get("markets", []))
            if isinstance(data, list):
                return [dict(row) for row in data]
        raise ValueError("invalid Gamma markets payload")


def _family_slug_suffix(family: str) -> str:
    return {
        "btc_updown_5m": "5m",
        "btc_updown_15m": "15m",
        "btc_updown_1h": "1h",
    }[family]
