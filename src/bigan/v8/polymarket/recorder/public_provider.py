"""Read-only provider contracts for real Polymarket corpus recording."""

from __future__ import annotations

from typing import Any, Protocol

from bigan.v8.polymarket.recorder.contracts import PolymarketRealCorpusRecorderConfig


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
