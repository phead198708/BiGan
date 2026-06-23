"""Read-only Polymarket raw corpus recorder."""

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
from bigan.v8.polymarket.recorder.operator import record_polymarket_real_corpus
from bigan.v8.polymarket.recorder.public_provider import (
    DEFAULT_POLYMARKET_CLOB_WS_MARKET_URL,
    PolymarketCLOBWebSocketOrderBookSource,
    PolymarketOrderBookSource,
    PolymarketPublicHTTPRealCorpusProvider,
    PolymarketRealCorpusPublicProvider,
    RealCorpusPublicProviderError,
)

__all__ = [
    "DEFAULT_BTC_FEATURE_CANDLE_SOURCE",
    "DEFAULT_OFFICIAL_SETTLEMENT_REFERENCE_SOURCE",
    "DEFAULT_RECORDER_ENDED_AT",
    "DEFAULT_RECORDER_STARTED_AT",
    "DEFAULT_SAMPLING_POLICY_SECONDS",
    "POLYMARKET_REAL_CORPUS_RECORDER_PHASE",
    "POLYMARKET_REAL_CORPUS_RECORDER_SCHEMA_VERSION",
    "DEFAULT_POLYMARKET_CLOB_WS_MARKET_URL",
    "PolymarketCLOBWebSocketOrderBookSource",
    "PolymarketOrderBookSource",
    "PolymarketRealCorpusRecorderConfig",
    "PolymarketRealCorpusRecorderResult",
    "PolymarketPublicHTTPRealCorpusProvider",
    "PolymarketRealCorpusPublicProvider",
    "RealCorpusPublicProviderError",
    "record_polymarket_real_corpus",
]
