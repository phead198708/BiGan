"""Contracts for Polymarket live-data paper-only runs."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from bigan.v8.polymarket.contracts import canonical_json_sha256, looks_like_sha256
from bigan.v8.polymarket.corpus import BTC_UPDOWN_MARKET_HORIZONS_MS

POLYMARKET_LIVE_SCHEMA_VERSION = "bigan-v8-polymarket-live-paper-v1"
POLYMARKET_LIVE_PHASE = "polymarket_live_paper"
DEFAULT_LIVE_PAPER_CREATED_AT = "1970-01-01T00:00:00Z"
DEFAULT_LIVE_PAPER_STARTED_AT = "1970-01-01T00:00:00Z"

LiveMode = Literal["dry-run", "gh-command"]
SettlementMode = Literal["resolved", "delayed"]
LiveOperatorRecommendation = Literal[
    "continue_paper_run",
    "stop_paper_run",
    "blocked_fail_closed",
    "await_settlement",
]
LiveOperatorStatus = Literal[
    "completed",
    "operator_stopped",
    "blocked_fail_closed",
    "awaiting_settlement",
]
LiveMarketStatus = Literal["open", "closed", "resolved", "settlement_pending"]
LiveOutcome = Literal["UP", "DOWN"]


class PolymarketLivePaperError(RuntimeError):
    """Raised when the live paper path cannot continue safely."""

    def __init__(self, message: str, *, reason_codes: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.reason_codes = reason_codes


@dataclass(frozen=True, slots=True)
class PolymarketLivePaperConfig:
    """Configuration for one Polymarket live-data, paper-only run."""

    run_id: str
    output_dir: Path | str
    repo_full_name: str = "phead198708/BiGan"
    issue_number: int = 134
    mode: LiveMode = "dry-run"
    mock_live: bool = True
    market_families: tuple[str, ...] = tuple(BTC_UPDOWN_MARKET_HORIZONS_MS)
    model_manifest: Path | str | None = None
    model_path: Path | str | None = None
    duration_seconds: int = 300
    poll_interval_seconds: int = 5
    summary_interval_seconds: int = 300
    stream_observability: bool = False
    status_interval_seconds: int = 15
    heartbeat_interval_seconds: int = 60
    flush_event_files: bool = False
    ev_threshold: float = 0.015
    min_confidence: float = 0.05
    max_paper_notional: float = 0.20
    fee_rate: float = 0.0002
    slippage_rate: float = 0.0005
    liquidity_impact_rate: float = 0.0001
    max_stale_orderbook_seconds: int = 60
    max_stale_reference_seconds: int = 60
    settlement_mode: SettlementMode = "resolved"
    settlement_wait_timeout_seconds: int = 600
    settlement_poll_interval_seconds: int = 15
    export_training_corpus: bool = False
    training_corpus_root: Path | str = Path("/Volumes/PHILIPS/v8")
    stop_requested: bool = False
    inject_missing_market_rule: bool = False
    inject_missing_token_book: bool = False
    inject_stale_orderbook: bool = False
    inject_stale_reference: bool = False
    inject_model_manifest_mismatch: bool = False
    created_at: str = DEFAULT_LIVE_PAPER_CREATED_AT
    started_at: str = DEFAULT_LIVE_PAPER_STARTED_AT
    overwrite_existing: bool = False
    paper_only: bool = True
    capital_at_risk: bool = False
    broker_exchange_write_enabled: bool = False
    live_exchange_write_enabled: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if not isinstance(self.output_dir, Path):
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.model_manifest is not None and not isinstance(self.model_manifest, Path):
            object.__setattr__(self, "model_manifest", Path(self.model_manifest))
        if self.model_path is not None and not isinstance(self.model_path, Path):
            object.__setattr__(self, "model_path", Path(self.model_path))
        if not isinstance(self.training_corpus_root, Path):
            object.__setattr__(
                self,
                "training_corpus_root",
                Path(self.training_corpus_root),
            )
        if self.mode not in ("dry-run", "gh-command"):
            raise ValueError("mode must be dry-run or gh-command")
        if not self.repo_full_name.strip() or "/" not in self.repo_full_name:
            raise ValueError("repo_full_name must be owner/repo")
        if self.issue_number <= 0:
            raise ValueError("issue_number must be positive")
        unsupported = set(self.market_families) - set(BTC_UPDOWN_MARKET_HORIZONS_MS)
        if unsupported:
            raise ValueError("unsupported market families: " + ", ".join(sorted(unsupported)))
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if self.poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        if self.summary_interval_seconds <= 0:
            raise ValueError("summary_interval_seconds must be positive")
        if self.status_interval_seconds <= 0:
            raise ValueError("status_interval_seconds must be positive")
        if self.heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat_interval_seconds must be positive")
        for field_name in (
            "ev_threshold",
            "min_confidence",
            "max_paper_notional",
            "fee_rate",
            "slippage_rate",
            "liquidity_impact_rate",
        ):
            value = float(getattr(self, field_name))
            if value < 0.0 or not math.isfinite(value):
                raise ValueError(f"{field_name} must be non-negative and finite")
        if self.max_stale_orderbook_seconds <= 0:
            raise ValueError("max_stale_orderbook_seconds must be positive")
        if self.max_stale_reference_seconds <= 0:
            raise ValueError("max_stale_reference_seconds must be positive")
        if self.settlement_mode not in ("resolved", "delayed"):
            raise ValueError("settlement_mode must be resolved or delayed")
        if self.settlement_wait_timeout_seconds < 0:
            raise ValueError("settlement_wait_timeout_seconds must be non-negative")
        if self.settlement_poll_interval_seconds <= 0:
            raise ValueError("settlement_poll_interval_seconds must be positive")
        if not self.created_at or not self.started_at:
            raise ValueError("created_at and started_at are required")
        _validate_full_safety_boundary(self)

    @property
    def run_dir(self) -> Path:
        return self.output_dir.expanduser().resolve() / self.run_id

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["market_families"] = list(self.market_families)
        payload["model_manifest"] = (
            None if self.model_manifest is None else str(self.model_manifest)
        )
        payload["model_path"] = None if self.model_path is None else str(self.model_path)
        payload["training_corpus_root"] = str(self.training_corpus_root)
        return payload


@dataclass(frozen=True, slots=True)
class PolymarketLiveMarket:
    """Read-only metadata for one live BTC UP/DOWN market."""

    market_id: str
    condition_id: str
    slug: str
    market_family: str
    horizon_ms: int
    market_start_ts: int
    market_end_ts: int
    settlement_ts: int
    up_token_id: str
    down_token_id: str
    reference_price_source: str
    settlement_rule: str
    reference_price_at_start: float
    status: LiveMarketStatus = "open"
    resolution_available: bool = False
    raw_market_sha256: str = ""
    read_only: bool = True
    write_capable: bool = False
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
            market_family=self.market_family,
            up_token_id=self.up_token_id,
            down_token_id=self.down_token_id,
            reference_price_source=self.reference_price_source,
            settlement_rule=self.settlement_rule,
        )
        if self.market_family not in BTC_UPDOWN_MARKET_HORIZONS_MS:
            raise ValueError("unsupported market_family")
        if self.horizon_ms != BTC_UPDOWN_MARKET_HORIZONS_MS[self.market_family]:
            raise ValueError("horizon_ms must match market_family")
        if self.market_end_ts <= self.market_start_ts:
            raise ValueError("market_end_ts must be after market_start_ts")
        if self.market_end_ts - self.market_start_ts != self.horizon_ms:
            raise ValueError("market window must match horizon")
        if self.settlement_ts < self.market_end_ts:
            raise ValueError("settlement_ts cannot precede market_end_ts")
        if self.up_token_id == self.down_token_id:
            raise ValueError("UP and DOWN token ids must differ")
        if self.reference_price_at_start <= 0.0:
            raise ValueError("reference_price_at_start must be positive")
        if self.status not in ("open", "closed", "resolved", "settlement_pending"):
            raise ValueError("unsupported live market status")
        if self.raw_market_sha256 and not looks_like_sha256(self.raw_market_sha256):
            raise ValueError("raw_market_sha256 must be SHA-256")
        _validate_readonly_boundary(self)
        if not self.raw_market_sha256:
            object.__setattr__(
                self,
                "raw_market_sha256",
                canonical_json_sha256(
                    {
                        "market_id": self.market_id,
                        "condition_id": self.condition_id,
                        "slug": self.slug,
                        "market_family": self.market_family,
                    }
                ),
            )

    def token_id_for_outcome(self, outcome: LiveOutcome) -> str:
        return self.up_token_id if outcome == "UP" else self.down_token_id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PolymarketLiveOrderBook:
    """Read-only Polymarket token order book snapshot."""

    market_id: str
    token_id: str
    outcome: LiveOutcome
    ts: int
    received_ts: int
    bid_price: float
    ask_price: float
    mid_price: float
    bid_size: float
    ask_size: float
    liquidity_depth: float
    source: str = "polymarket_public"
    read_only: bool = True
    write_capable: bool = False
    paper_only: bool = True
    capital_at_risk: bool = False
    broker_exchange_write_enabled: bool = False
    live_exchange_write_enabled: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False

    def __post_init__(self) -> None:
        _require_non_empty(
            market_id=self.market_id,
            token_id=self.token_id,
            outcome=self.outcome,
            source=self.source,
        )
        if self.outcome not in ("UP", "DOWN"):
            raise ValueError("outcome must be UP or DOWN")
        if self.ts < 0 or self.received_ts < self.ts:
            raise ValueError("received_ts must be >= ts")
        if not (0.0 < self.bid_price <= self.ask_price <= 1.0):
            raise ValueError("bid/ask prices must be ordered inside (0, 1]")
        if self.mid_price <= 0.0 or not math.isfinite(self.mid_price):
            raise ValueError("mid_price must be positive and finite")
        for field_name in ("bid_size", "ask_size", "liquidity_depth"):
            value = float(getattr(self, field_name))
            if value < 0.0 or not math.isfinite(value):
                raise ValueError(f"{field_name} must be non-negative and finite")
        _validate_readonly_boundary(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PolymarketLiveTrade:
    """Read-only public trade observation."""

    market_id: str
    token_id: str
    outcome: LiveOutcome
    ts: int
    price: float
    size: float
    side: str
    source: str = "polymarket_public"
    read_only: bool = True
    write_capable: bool = False
    paper_only: bool = True
    capital_at_risk: bool = False
    broker_exchange_write_enabled: bool = False
    live_exchange_write_enabled: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False

    def __post_init__(self) -> None:
        _require_non_empty(
            market_id=self.market_id,
            token_id=self.token_id,
            outcome=self.outcome,
            side=self.side,
        )
        if self.outcome not in ("UP", "DOWN"):
            raise ValueError("outcome must be UP or DOWN")
        if self.ts < 0:
            raise ValueError("ts must be non-negative")
        if not 0.0 < self.price <= 1.0:
            raise ValueError("price must be inside (0, 1]")
        if self.size < 0.0:
            raise ValueError("size must be non-negative")
        _validate_readonly_boundary(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BinanceBTCReferenceTick:
    """Read-only BTCUSDT reference tick."""

    ts: int
    received_ts: int
    bid_price: float
    ask_price: float
    mid_price: float
    last_price: float
    source: str = "binance_public"
    instrument_id: str = "BTCUSDT"
    read_only: bool = True
    write_capable: bool = False
    paper_only: bool = True
    capital_at_risk: bool = False
    broker_exchange_write_enabled: bool = False
    live_exchange_write_enabled: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False

    def __post_init__(self) -> None:
        _require_non_empty(source=self.source, instrument_id=self.instrument_id)
        if self.ts < 0 or self.received_ts < self.ts:
            raise ValueError("received_ts must be >= ts")
        if not (0.0 < self.bid_price <= self.ask_price):
            raise ValueError("bid/ask prices must be positive and ordered")
        if self.mid_price <= 0.0 or self.last_price <= 0.0:
            raise ValueError("mid/last prices must be positive")
        _validate_readonly_boundary(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BinanceBTCCandle:
    """Read-only BTCUSDT candle used for market resolution."""

    market_id: str
    market_family: str
    open_ts: int
    close_ts: int
    open_price: float
    close_price: float
    high_price: float
    low_price: float
    source: str = "binance_public"
    instrument_id: str = "BTCUSDT"
    read_only: bool = True
    write_capable: bool = False
    paper_only: bool = True
    capital_at_risk: bool = False
    broker_exchange_write_enabled: bool = False
    live_exchange_write_enabled: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False

    def __post_init__(self) -> None:
        _require_non_empty(
            market_id=self.market_id,
            market_family=self.market_family,
            source=self.source,
            instrument_id=self.instrument_id,
        )
        if self.market_family not in BTC_UPDOWN_MARKET_HORIZONS_MS:
            raise ValueError("unsupported market_family")
        if self.close_ts <= self.open_ts:
            raise ValueError("close_ts must be after open_ts")
        for field_name in ("open_price", "close_price", "high_price", "low_price"):
            value = float(getattr(self, field_name))
            if value <= 0.0 or not math.isfinite(value):
                raise ValueError(f"{field_name} must be positive and finite")
        _validate_readonly_boundary(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PolymarketLivePaperResult:
    """Output handles for one live paper run."""

    run_dir: Path
    artifact_paths: dict[str, Path]
    operator_manifest: dict[str, Any]
    observability_report: dict[str, Any]
    pnl_breakdown: dict[str, Any]


def safety_fields() -> dict[str, bool]:
    return {
        "paper_only": True,
        "capital_at_risk": False,
        "broker_exchange_write_enabled": False,
        "live_exchange_write_enabled": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }


def compact_safety_fields() -> dict[str, bool]:
    return {
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }


def _validate_full_safety_boundary(payload: Any) -> None:
    for field_name, expected in safety_fields().items():
        if getattr(payload, field_name) is not expected:
            raise ValueError(f"{field_name} must be {str(expected).lower()}")


def _validate_readonly_boundary(payload: Any) -> None:
    if payload.read_only is not True:
        raise ValueError("live feed artifacts must be read-only")
    if payload.write_capable is not False:
        raise ValueError("write-capable live feed artifacts are forbidden")
    _validate_full_safety_boundary(payload)


def _require_non_empty(**values: str) -> None:
    for field_name, value in values.items():
        if not str(value).strip():
            raise ValueError(f"{field_name} is required")
