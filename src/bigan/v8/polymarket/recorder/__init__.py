"""Read-only Polymarket raw corpus recorder."""

from bigan.v8.polymarket.recorder.async_settlement import (
    ASYNC_SETTLEMENT_SCHEMA_VERSION,
    PENDING_CAPTURE_PHASE,
    PENDING_FINALIZATION_PHASE,
    PendingRoundCaptureResult,
    PendingRoundFinalizationResult,
    capture_polymarket_pending_round,
    finalize_polymarket_pending_round,
)
from bigan.v8.polymarket.recorder.chainlink_rtds import (
    CHAINLINK_RTDS_COLLECTION_REPORT_FILENAME,
    CHAINLINK_RTDS_CORPUS_FILENAME,
    CHAINLINK_RTDS_CORPUS_MANIFEST_FILENAME,
    CHAINLINK_RTDS_RAW_FILENAME,
    DEFAULT_POLYMARKET_RTDS_URL,
    PolymarketChainlinkRTDSCollector,
)
from bigan.v8.polymarket.recorder.contracts import (
    DEFAULT_BTC_FEATURE_CANDLE_SOURCE,
    DEFAULT_OFFICIAL_SETTLEMENT_REFERENCE_SOURCE,
    DEFAULT_RECORDER_ENDED_AT,
    DEFAULT_RECORDER_STARTED_AT,
    DEFAULT_SAMPLING_POLICY_SECONDS,
    POLYMARKET_REAL_CORPUS_RECORDER_PHASE,
    POLYMARKET_REAL_CORPUS_RECORDER_SCHEMA_VERSION,
    PolymarketRealCorpusRecorderConfig,
    PolymarketRealCorpusRecorderResult,
)
from bigan.v8.polymarket.recorder.market_identity_cache import (
    GAMMA_MARKET_IDENTITY_CACHE_FALLBACK_SOURCE_TYPE,
    GAMMA_MARKET_IDENTITY_CACHE_SCHEMA_VERSION,
    GAMMA_MARKET_IDENTITY_CACHE_SOURCE_TYPE,
    GammaMarketIdentityCache,
    GammaMarketIdentityCacheError,
)
from bigan.v8.polymarket.recorder.operator import record_polymarket_real_corpus
from bigan.v8.polymarket.recorder.public_provider import (
    DEFAULT_POLYMARKET_CLOB_WS_MARKET_URL,
    PolymarketCLOBWebSocketOrderBookSource,
    PolymarketOrderBookSource,
    PolymarketPublicHTTPRealCorpusProvider,
    PolymarketRealCorpusPublicProvider,
    RealCorpusPublicProviderError,
)
from bigan.v8.polymarket.recorder.resolution import (
    normalize_resolution_for_settlement,
)

__all__ = [
    "DEFAULT_BTC_FEATURE_CANDLE_SOURCE",
    "DEFAULT_OFFICIAL_SETTLEMENT_REFERENCE_SOURCE",
    "DEFAULT_RECORDER_ENDED_AT",
    "DEFAULT_RECORDER_STARTED_AT",
    "DEFAULT_SAMPLING_POLICY_SECONDS",
    "ASYNC_SETTLEMENT_SCHEMA_VERSION",
    "PENDING_CAPTURE_PHASE",
    "PENDING_FINALIZATION_PHASE",
    "POLYMARKET_REAL_CORPUS_RECORDER_PHASE",
    "POLYMARKET_REAL_CORPUS_RECORDER_SCHEMA_VERSION",
    "DEFAULT_POLYMARKET_CLOB_WS_MARKET_URL",
    "DEFAULT_POLYMARKET_RTDS_URL",
    "CHAINLINK_RTDS_COLLECTION_REPORT_FILENAME",
    "CHAINLINK_RTDS_CORPUS_FILENAME",
    "CHAINLINK_RTDS_CORPUS_MANIFEST_FILENAME",
    "CHAINLINK_RTDS_RAW_FILENAME",
    "PendingRoundCaptureResult",
    "PendingRoundFinalizationResult",
    "PolymarketCLOBWebSocketOrderBookSource",
    "PolymarketChainlinkRTDSCollector",
    "PolymarketOrderBookSource",
    "PolymarketRealCorpusRecorderConfig",
    "PolymarketRealCorpusRecorderResult",
    "GAMMA_MARKET_IDENTITY_CACHE_FALLBACK_SOURCE_TYPE",
    "GAMMA_MARKET_IDENTITY_CACHE_SCHEMA_VERSION",
    "GAMMA_MARKET_IDENTITY_CACHE_SOURCE_TYPE",
    "GammaMarketIdentityCache",
    "GammaMarketIdentityCacheError",
    "PolymarketPublicHTTPRealCorpusProvider",
    "PolymarketRealCorpusPublicProvider",
    "RealCorpusPublicProviderError",
    "capture_polymarket_pending_round",
    "finalize_polymarket_pending_round",
    "normalize_resolution_for_settlement",
    "record_polymarket_real_corpus",
]
