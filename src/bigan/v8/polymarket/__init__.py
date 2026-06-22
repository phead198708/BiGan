"""Polymarket-specific adapter layer for v8 paper-only trading."""

from bigan.v8.polymarket.contracts import (
    POLYMARKET_ADAPTER_SCHEMA_VERSION,
    POLYMARKET_BTC15M_HORIZON_MS,
    POLYMARKET_BTC15M_MARKET_FAMILY,
    POLYMARKET_SOURCE,
    PolymarketAdapterError,
    PolymarketBinaryDecision,
    PolymarketBinaryMarket,
    PolymarketFeatureRow,
    PolymarketLabelRow,
    PolymarketTokenSnapshot,
    canonical_json_sha256,
    looks_like_sha256,
)
from bigan.v8.polymarket.features import build_polymarket_feature_rows
from bigan.v8.polymarket.labels import build_polymarket_label_rows
from bigan.v8.polymarket.market_adapter import (
    DEFAULT_POLYMARKET_ADAPTER_CREATED_AT,
    POLYMARKET_ADAPTER_PHASE,
    PolymarketAdapterRunConfig,
    PolymarketAdapterRunResult,
    normalize_btc15m_binary_market,
    normalize_token_snapshots,
    run_polymarket_btc15m_paper_pipeline,
    synthetic_btc15m_market_payload,
    synthetic_btc_market_rows,
    synthetic_token_snapshot_rows,
)
from bigan.v8.polymarket.paper_decision import (
    PolymarketPolicySignal,
    build_polymarket_paper_decisions,
    polymarket_decisions_to_phase4,
)

__all__ = [
    "DEFAULT_POLYMARKET_ADAPTER_CREATED_AT",
    "POLYMARKET_ADAPTER_PHASE",
    "POLYMARKET_ADAPTER_SCHEMA_VERSION",
    "POLYMARKET_BTC15M_HORIZON_MS",
    "POLYMARKET_BTC15M_MARKET_FAMILY",
    "POLYMARKET_SOURCE",
    "PolymarketAdapterError",
    "PolymarketAdapterRunConfig",
    "PolymarketAdapterRunResult",
    "PolymarketBinaryDecision",
    "PolymarketBinaryMarket",
    "PolymarketFeatureRow",
    "PolymarketLabelRow",
    "PolymarketPolicySignal",
    "PolymarketTokenSnapshot",
    "build_polymarket_feature_rows",
    "build_polymarket_label_rows",
    "build_polymarket_paper_decisions",
    "canonical_json_sha256",
    "looks_like_sha256",
    "normalize_btc15m_binary_market",
    "normalize_token_snapshots",
    "polymarket_decisions_to_phase4",
    "run_polymarket_btc15m_paper_pipeline",
    "synthetic_btc15m_market_payload",
    "synthetic_btc_market_rows",
    "synthetic_token_snapshot_rows",
]
