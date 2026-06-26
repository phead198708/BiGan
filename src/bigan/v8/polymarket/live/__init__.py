"""Polymarket live-data paper-only operator package."""

from bigan.v8.polymarket.live.binance_reference_feed import (
    BinanceBTCReferenceHTTPFeed,
    MockBinanceBTCReferenceFeed,
)
from bigan.v8.polymarket.live.contracts import (
    DEFAULT_LIVE_PAPER_CREATED_AT,
    DEFAULT_LIVE_PAPER_STARTED_AT,
    POLYMARKET_LIVE_PHASE,
    POLYMARKET_LIVE_SCHEMA_VERSION,
    BinanceBTCCandle,
    BinanceBTCReferenceTick,
    PolymarketLiveMarket,
    PolymarketLiveOrderBook,
    PolymarketLivePaperConfig,
    PolymarketLivePaperError,
    PolymarketLivePaperResult,
    PolymarketLiveTrade,
)
from bigan.v8.polymarket.live.operator import (
    finalize_polymarket_round_artifacts,
    run_polymarket_live_paper,
    write_polymarket_round_lifecycle_indexes,
)
from bigan.v8.polymarket.live.polymarket_feed import (
    MockPolymarketLiveFeed,
    PolymarketHTTPReadOnlyFeed,
)

__all__ = [
    "DEFAULT_LIVE_PAPER_CREATED_AT",
    "DEFAULT_LIVE_PAPER_STARTED_AT",
    "POLYMARKET_LIVE_PHASE",
    "POLYMARKET_LIVE_SCHEMA_VERSION",
    "BinanceBTCCandle",
    "BinanceBTCReferenceHTTPFeed",
    "BinanceBTCReferenceTick",
    "MockBinanceBTCReferenceFeed",
    "MockPolymarketLiveFeed",
    "PolymarketHTTPReadOnlyFeed",
    "PolymarketLiveMarket",
    "PolymarketLiveOrderBook",
    "PolymarketLivePaperConfig",
    "PolymarketLivePaperError",
    "PolymarketLivePaperResult",
    "PolymarketLiveTrade",
    "finalize_polymarket_round_artifacts",
    "run_polymarket_live_paper",
    "write_polymarket_round_lifecycle_indexes",
]
