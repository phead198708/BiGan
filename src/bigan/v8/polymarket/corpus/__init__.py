"""Historical corpus builder for v8 Polymarket BTC UP/DOWN markets."""

from bigan.v8.polymarket.corpus.aggregate import (
    POLYMARKET_REAL_CORPUS_AGGREGATE_SCHEMA_VERSION,
    SPARSE_THEORETICAL_TRAINING_EXCLUSION_REASON,
    PolymarketRealCorpusAggregateConfig,
    PolymarketRealCorpusAggregateResult,
    build_polymarket_real_corpus_aggregate,
)
from bigan.v8.polymarket.corpus.builder import (
    build_polymarket_btc_corpus,
    write_deterministic_polymarket_corpus_fixtures,
)
from bigan.v8.polymarket.corpus.contracts import (
    BTC_UPDOWN_MARKET_HORIZONS_MS,
    DEFAULT_CORPUS_CREATED_AT,
    NORMALIZED_CORPUS_FILENAMES,
    POLYMARKET_CORPUS_PHASE,
    POLYMARKET_CORPUS_SCHEMA_VERSION,
    POLYMARKET_SELL_BEFORE_CLOSE_LABEL_REDESIGN_REPORT_SCHEMA_VERSION,
    POLYMARKET_SELL_BEFORE_CLOSE_LABEL_SCHEMA_VERSION,
    RAW_CORPUS_FILENAMES,
    BinanceBTCCandle,
    PolymarketCorpusBookSnapshot,
    PolymarketCorpusBuildConfig,
    PolymarketCorpusBuildResult,
    PolymarketCorpusFeatureRow,
    PolymarketCorpusLabelRow,
    PolymarketCorpusMarket,
    PolymarketCorpusResolutionEvent,
    PolymarketCorpusSplit,
    PolymarketCorpusTrade,
)
from bigan.v8.polymarket.corpus.features import build_polymarket_corpus_feature_rows
from bigan.v8.polymarket.corpus.labels import build_polymarket_corpus_label_rows
from bigan.v8.polymarket.corpus.live_converter import (
    LiveSignalCorpusConversionConfig,
    LiveSignalCorpusConversionResult,
    convert_live_signals_to_phase2_corpus,
)
from bigan.v8.polymarket.corpus.splits import build_polymarket_train_shadow_split

__all__ = [
    "BTC_UPDOWN_MARKET_HORIZONS_MS",
    "DEFAULT_CORPUS_CREATED_AT",
    "NORMALIZED_CORPUS_FILENAMES",
    "POLYMARKET_CORPUS_PHASE",
    "POLYMARKET_CORPUS_SCHEMA_VERSION",
    "POLYMARKET_REAL_CORPUS_AGGREGATE_SCHEMA_VERSION",
    "POLYMARKET_SELL_BEFORE_CLOSE_LABEL_REDESIGN_REPORT_SCHEMA_VERSION",
    "POLYMARKET_SELL_BEFORE_CLOSE_LABEL_SCHEMA_VERSION",
    "RAW_CORPUS_FILENAMES",
    "SPARSE_THEORETICAL_TRAINING_EXCLUSION_REASON",
    "BinanceBTCCandle",
    "PolymarketCorpusBookSnapshot",
    "PolymarketCorpusBuildConfig",
    "PolymarketCorpusBuildResult",
    "PolymarketCorpusFeatureRow",
    "PolymarketCorpusLabelRow",
    "PolymarketCorpusMarket",
    "PolymarketCorpusResolutionEvent",
    "PolymarketCorpusSplit",
    "PolymarketCorpusTrade",
    "PolymarketRealCorpusAggregateConfig",
    "PolymarketRealCorpusAggregateResult",
    "LiveSignalCorpusConversionConfig",
    "LiveSignalCorpusConversionResult",
    "build_polymarket_btc_corpus",
    "build_polymarket_corpus_feature_rows",
    "build_polymarket_corpus_label_rows",
    "build_polymarket_real_corpus_aggregate",
    "build_polymarket_train_shadow_split",
    "convert_live_signals_to_phase2_corpus",
    "write_deterministic_polymarket_corpus_fixtures",
]
