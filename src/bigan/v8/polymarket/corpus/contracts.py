"""Contracts for deterministic Polymarket BTC UP/DOWN corpus artifacts."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from bigan.v8.polymarket.contracts import canonical_json_sha256, looks_like_sha256

POLYMARKET_CORPUS_SCHEMA_VERSION = "bigan-v8-polymarket-corpus-v1"
POLYMARKET_CORPUS_PHASE = "polymarket_historical_corpus"
DEFAULT_CORPUS_CREATED_AT = "1970-01-01T00:00:00Z"

BTC_UPDOWN_MARKET_HORIZONS_MS: dict[str, int] = {
    "btc_updown_5m": 5 * 60 * 1000,
    "btc_updown_15m": 15 * 60 * 1000,
    "btc_updown_1h": 60 * 60 * 1000,
}

RAW_CORPUS_FILENAMES: tuple[str, ...] = (
    "raw_polymarket_markets.jsonl",
    "raw_polymarket_orderbooks.jsonl",
    "raw_polymarket_trades.jsonl",
    "raw_binance_btcusdt_klines.jsonl",
    "raw_polymarket_resolutions.jsonl",
)

NORMALIZED_CORPUS_FILENAMES: tuple[str, ...] = (
    "polymarket_market_rules.jsonl",
    "polymarket_market_metadata.jsonl",
    "polymarket_token_book_snapshots.jsonl",
    "polymarket_token_trades.jsonl",
    "polymarket_btc_reference_candles.jsonl",
    "polymarket_resolution_events.jsonl",
    "polymarket_feature_rows.jsonl",
    "polymarket_label_rows.jsonl",
    "polymarket_train_shadow_split.json",
    "polymarket_corpus_summary.json",
)

CorpusMarketFamily = Literal["btc_updown_5m", "btc_updown_15m", "btc_updown_1h"]
CorpusOutcome = Literal["UP", "DOWN"]
CorpusLabelOutcome = Literal["UP", "DOWN", "NONE"]
CorpusLabelAction = Literal[
    "NO_TRADE",
    "BUY_UP_HOLD_TO_SETTLEMENT",
    "BUY_DOWN_HOLD_TO_SETTLEMENT",
    "BUY_UP_SELL_BEFORE_CLOSE",
    "BUY_DOWN_SELL_BEFORE_CLOSE",
]
CorpusResolutionStatus = Literal["normal", "unknown_50_50"]


@dataclass(frozen=True, slots=True)
class PolymarketCorpusBuildConfig:
    """Configuration for a deterministic local corpus build."""

    input_dir: Path | str
    output_dir: Path | str
    created_at: str = DEFAULT_CORPUS_CREATED_AT
    market_families: tuple[CorpusMarketFamily, ...] = (
        "btc_updown_5m",
        "btc_updown_15m",
        "btc_updown_1h",
    )
    sample_interval_seconds: dict[str, int] | None = None
    min_time_to_close_seconds: int = 0
    max_time_to_close_seconds: int | None = None
    include_trade_labels: bool = True
    include_settlement_labels: bool = True
    train_fraction: float = 0.67
    overwrite_existing: bool = False
    paper_only: bool = True
    capital_at_risk: bool = False
    broker_exchange_write_enabled: bool = False
    live_exchange_write_enabled: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.input_dir, Path):
            object.__setattr__(self, "input_dir", Path(self.input_dir))
        if not isinstance(self.output_dir, Path):
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not self.created_at:
            raise ValueError("created_at is required")
        if not self.market_families:
            raise ValueError("market_families must not be empty")
        unsupported = set(self.market_families) - set(BTC_UPDOWN_MARKET_HORIZONS_MS)
        if unsupported:
            raise ValueError("unsupported market families: " + ", ".join(sorted(unsupported)))
        intervals = self.resolved_sample_intervals()
        for family, seconds in intervals.items():
            if family not in BTC_UPDOWN_MARKET_HORIZONS_MS:
                raise ValueError(f"unsupported sample interval family: {family}")
            if seconds <= 0:
                raise ValueError("sample intervals must be positive")
        if self.min_time_to_close_seconds < 0:
            raise ValueError("min_time_to_close_seconds must be non-negative")
        if (
            self.max_time_to_close_seconds is not None
            and self.max_time_to_close_seconds < self.min_time_to_close_seconds
        ):
            raise ValueError("max_time_to_close_seconds must be >= min_time_to_close")
        if not (0.0 < self.train_fraction < 1.0):
            raise ValueError("train_fraction must be in (0, 1)")
        if not self.include_trade_labels and not self.include_settlement_labels:
            raise ValueError("at least one label family must be enabled")
        _validate_safety_boundary(self)

    def resolved_sample_intervals(self) -> dict[str, int]:
        default = {
            "btc_updown_5m": 60,
            "btc_updown_15m": 300,
            "btc_updown_1h": 900,
        }
        if self.sample_interval_seconds:
            default.update({str(k): int(v) for k, v in self.sample_interval_seconds.items()})
        return default

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["input_dir"] = str(self.input_dir)
        payload["output_dir"] = str(self.output_dir)
        payload["market_families"] = list(self.market_families)
        payload["sample_interval_seconds"] = self.resolved_sample_intervals()
        return payload

    def to_manifest_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("input_dir", None)
        payload.pop("output_dir", None)
        return payload


@dataclass(frozen=True, slots=True)
class PolymarketCorpusMarket:
    market_id: str
    condition_id: str
    slug: str
    market_family: CorpusMarketFamily
    horizon_ms: int
    market_start_ts: int
    market_end_ts: int
    settlement_ts: int
    up_token_id: str
    down_token_id: str
    reference_price_source: str
    settlement_rule: str
    raw_market_sha256: str
    paper_only: bool = True
    capital_at_risk: bool = False
    broker_exchange_write_enabled: bool = False
    live_exchange_write_enabled: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False

    def __post_init__(self) -> None:
        _require_non_empty(
            market_id=self.market_id,
            condition_id=self.condition_id,
            slug=self.slug,
            up_token_id=self.up_token_id,
            down_token_id=self.down_token_id,
            reference_price_source=self.reference_price_source,
            settlement_rule=self.settlement_rule,
            raw_market_sha256=self.raw_market_sha256,
        )
        if self.market_family not in BTC_UPDOWN_MARKET_HORIZONS_MS:
            raise ValueError("unsupported market_family")
        if self.horizon_ms != BTC_UPDOWN_MARKET_HORIZONS_MS[self.market_family]:
            raise ValueError("horizon_ms must match market_family")
        if self.market_end_ts - self.market_start_ts != self.horizon_ms:
            raise ValueError("market window must match horizon_ms")
        if self.settlement_ts < self.market_end_ts:
            raise ValueError("settlement_ts cannot precede market_end_ts")
        if self.up_token_id == self.down_token_id:
            raise ValueError("UP and DOWN token ids must differ")
        if not looks_like_sha256(self.raw_market_sha256):
            raise ValueError("raw_market_sha256 must be SHA-256")
        _validate_safety_boundary(self)

    def token_id_for_outcome(self, outcome: CorpusOutcome) -> str:
        return self.up_token_id if outcome == "UP" else self.down_token_id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PolymarketCorpusBookSnapshot:
    market_id: str
    token_id: str
    outcome: CorpusOutcome
    ts: int
    available_at_ts: int
    bid_price: float
    ask_price: float
    mid_price: float
    bid_size: float
    ask_size: float
    liquidity_depth: float
    source: str = "polymarket"
    paper_only: bool = True
    capital_at_risk: bool = False
    broker_exchange_write_enabled: bool = False
    live_exchange_write_enabled: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False

    def __post_init__(self) -> None:
        _require_non_empty(market_id=self.market_id, token_id=self.token_id, source=self.source)
        if self.outcome not in ("UP", "DOWN"):
            raise ValueError("outcome must be UP or DOWN")
        if self.available_at_ts < self.ts:
            raise ValueError("available_at_ts cannot be earlier than ts")
        if not (0.0 < self.bid_price <= self.ask_price <= 1.0):
            raise ValueError("token bid/ask must be in (0, 1] and ordered")
        if self.mid_price <= 0.0 or self.mid_price > 1.0:
            raise ValueError("mid_price must be in (0, 1]")
        if self.bid_size < 0.0 or self.ask_size < 0.0 or self.liquidity_depth < 0.0:
            raise ValueError("book sizes cannot be negative")
        _validate_safety_boundary(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PolymarketCorpusTrade:
    market_id: str
    token_id: str
    outcome: CorpusOutcome
    ts: int
    available_at_ts: int
    price: float
    size: float
    side: str
    source: str = "polymarket"
    paper_only: bool = True
    capital_at_risk: bool = False
    broker_exchange_write_enabled: bool = False
    live_exchange_write_enabled: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False

    def __post_init__(self) -> None:
        _require_non_empty(market_id=self.market_id, token_id=self.token_id, side=self.side)
        if self.outcome not in ("UP", "DOWN"):
            raise ValueError("outcome must be UP or DOWN")
        if self.available_at_ts < self.ts:
            raise ValueError("available_at_ts cannot be earlier than ts")
        if not (0.0 < self.price <= 1.0):
            raise ValueError("trade price must be in (0, 1]")
        if self.size < 0.0 or not math.isfinite(self.size):
            raise ValueError("trade size must be non-negative and finite")
        _validate_safety_boundary(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BinanceBTCCandle:
    ts: int
    available_at_ts: int
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    timeframe_ms: int
    source: str = "binance_btcusdt"

    def __post_init__(self) -> None:
        if self.available_at_ts < self.ts:
            raise ValueError("available_at_ts cannot be earlier than ts")
        for field_name in ("open_price", "high_price", "low_price", "close_price"):
            value = float(getattr(self, field_name))
            if value <= 0.0 or not math.isfinite(value):
                raise ValueError(f"{field_name} must be positive and finite")
        if self.high_price < max(self.open_price, self.close_price):
            raise ValueError("high_price must cover open/close")
        if self.low_price > min(self.open_price, self.close_price):
            raise ValueError("low_price must cover open/close")
        if self.volume < 0.0 or self.timeframe_ms <= 0:
            raise ValueError("volume/timeframe must be valid")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PolymarketCorpusResolutionEvent:
    market_id: str
    condition_id: str
    slug: str
    market_family: CorpusMarketFamily
    reference_price_start: float
    reference_price_end: float
    resolution_status: CorpusResolutionStatus
    resolved_outcome: str
    payout_up: float
    payout_down: float
    resolution_rule_sha256: str
    raw_resolution_sha256: str
    paper_only: bool = True
    capital_at_risk: bool = False
    broker_exchange_write_enabled: bool = False
    live_exchange_write_enabled: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False

    def __post_init__(self) -> None:
        _require_non_empty(
            market_id=self.market_id,
            condition_id=self.condition_id,
            slug=self.slug,
            resolution_rule_sha256=self.resolution_rule_sha256,
            raw_resolution_sha256=self.raw_resolution_sha256,
        )
        if self.resolution_status not in ("normal", "unknown_50_50"):
            raise ValueError("unsupported resolution_status")
        if self.resolved_outcome not in ("UP", "DOWN", "UNKNOWN_50_50"):
            raise ValueError("unsupported resolved_outcome")
        if self.reference_price_start <= 0.0 or self.reference_price_end <= 0.0:
            raise ValueError("reference prices must be positive")
        if not looks_like_sha256(self.resolution_rule_sha256):
            raise ValueError("resolution_rule_sha256 must be SHA-256")
        if not looks_like_sha256(self.raw_resolution_sha256):
            raise ValueError("raw_resolution_sha256 must be SHA-256")
        _validate_safety_boundary(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PolymarketCorpusFeatureRow:
    market_id: str
    condition_id: str
    slug: str
    market_family: CorpusMarketFamily
    horizon_ms: int
    decision_ts: int
    feature_cutoff_ts: int
    max_input_ts: int
    available_at_ts: int
    features: dict[str, float | int | str | None]
    feature_provenance: dict[str, dict[str, int | str | None]]
    paper_only: bool = True
    capital_at_risk: bool = False
    broker_exchange_write_enabled: bool = False
    live_exchange_write_enabled: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False

    def __post_init__(self) -> None:
        _require_non_empty(market_id=self.market_id, condition_id=self.condition_id, slug=self.slug)
        if self.feature_cutoff_ts > self.decision_ts:
            raise ValueError("feature_cutoff_ts cannot exceed decision_ts")
        if self.max_input_ts > self.decision_ts:
            raise ValueError("max_input_ts cannot exceed decision_ts")
        if self.available_at_ts > self.decision_ts:
            raise ValueError("available_at_ts cannot exceed decision_ts")
        if set(self.features) - set(self.feature_provenance):
            raise ValueError("every feature must have provenance")
        _validate_safety_boundary(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PolymarketCorpusLabelRow:
    market_id: str
    condition_id: str
    slug: str
    market_family: CorpusMarketFamily
    horizon_ms: int
    decision_ts: int
    action: CorpusLabelAction
    outcome: CorpusLabelOutcome
    entry_bid: float
    entry_ask: float
    entry_mid: float
    exit_bid: float
    exit_ask: float
    settlement_payout: float
    realized_trade_return: float
    settlement_return: float
    total_net_return: float
    fees: float
    slippage: float
    liquidity_impact: float
    is_positive: bool
    resolved_outcome: str
    resolution_status: CorpusResolutionStatus
    comparator: str
    tie_breaker: str
    resolution_rule_sha256: str
    raw_resolution_sha256: str
    paper_only: bool = True
    capital_at_risk: bool = False
    broker_exchange_write_enabled: bool = False
    live_exchange_write_enabled: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False

    def __post_init__(self) -> None:
        _require_non_empty(market_id=self.market_id, condition_id=self.condition_id, slug=self.slug)
        if self.action not in (
            "NO_TRADE",
            "BUY_UP_HOLD_TO_SETTLEMENT",
            "BUY_DOWN_HOLD_TO_SETTLEMENT",
            "BUY_UP_SELL_BEFORE_CLOSE",
            "BUY_DOWN_SELL_BEFORE_CLOSE",
        ):
            raise ValueError("unsupported label action")
        if self.outcome not in ("UP", "DOWN", "NONE"):
            raise ValueError("unsupported label outcome")
        for field_name in (
            "entry_bid",
            "entry_ask",
            "entry_mid",
            "exit_bid",
            "exit_ask",
            "settlement_payout",
            "fees",
            "slippage",
            "liquidity_impact",
        ):
            value = float(getattr(self, field_name))
            if value < 0.0 or not math.isfinite(value):
                raise ValueError(f"{field_name} must be non-negative and finite")
        if self.action != "NO_TRADE" and self.entry_ask <= 0.0:
            raise ValueError("tradable labels require positive entry_ask")
        if self.action.endswith("SELL_BEFORE_CLOSE") and self.exit_bid <= 0.0:
            raise ValueError("sell labels require positive exit_bid")
        if self.is_positive != (self.total_net_return > 0.0):
            raise ValueError("is_positive must equal total_net_return > 0")
        if not looks_like_sha256(self.resolution_rule_sha256):
            raise ValueError("resolution_rule_sha256 must be SHA-256")
        if not looks_like_sha256(self.raw_resolution_sha256):
            raise ValueError("raw_resolution_sha256 must be SHA-256")
        _validate_safety_boundary(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PolymarketCorpusSplit:
    split_ts: int
    split_hash: str
    train_label_count: int
    shadow_label_count: int
    max_train_decision_ts: int
    min_shadow_decision_ts: int
    train_dataset_hash: str
    shadow_dataset_hash: str
    paper_only: bool = True
    capital_at_risk: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False

    def __post_init__(self) -> None:
        if self.train_label_count <= 0 or self.shadow_label_count <= 0:
            raise ValueError("train and shadow splits must both be non-empty")
        if self.max_train_decision_ts >= self.min_shadow_decision_ts:
            raise ValueError("temporal split must be leak-free")
        if self.max_train_decision_ts >= self.split_ts or self.min_shadow_decision_ts < self.split_ts:
            raise ValueError("split_ts must separate train and shadow")
        for field_name in ("split_hash", "train_dataset_hash", "shadow_dataset_hash"):
            if not looks_like_sha256(str(getattr(self, field_name))):
                raise ValueError(f"{field_name} must be SHA-256")
        if self.paper_only is not True or self.capital_at_risk is not False:
            raise ValueError("split must preserve paper-only boundary")
        if self.polymarket_write_enabled is not False or self.wallet_signing_enabled is not False:
            raise ValueError("split cannot enable Polymarket writes or wallet signing")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PolymarketCorpusBuildResult:
    output_dir: Path
    artifact_paths: dict[str, Path]
    artifact_hashes: dict[str, str]
    manifest: dict[str, Any]
    summary: dict[str, Any]


def safety_fields() -> dict[str, bool]:
    return {
        "paper_only": True,
        "capital_at_risk": False,
        "broker_exchange_write_enabled": False,
        "live_exchange_write_enabled": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }


def stable_hash(payload: Any) -> str:
    return canonical_json_sha256(payload)


def _require_non_empty(**values: str) -> None:
    for field_name, value in values.items():
        if not str(value).strip():
            raise ValueError(f"{field_name} is required")


def _validate_safety_boundary(payload: Any) -> None:
    for field_name, expected in safety_fields().items():
        if getattr(payload, field_name) is not expected:
            raise ValueError(f"{field_name} must be {str(expected).lower()}")
