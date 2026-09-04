"""Strict, paper-safe configuration for the long-running operator."""

from __future__ import annotations

import hashlib
import json
import math
import re
import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

DEFAULT_GAMMA_MARKETS_ENDPOINT = "https://gamma-api.polymarket.com/markets"
DEFAULT_BINANCE_DEPTH_ENDPOINT = "https://api.binance.com/api/v3/depth"
DEFAULT_BINANCE_WS_URL = "wss://stream.binance.com:9443/ws"
DEFAULT_POLYMARKET_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
DEFAULT_CHAINLINK_WS_URL = "wss://ws-live-data.polymarket.com"

_DANGEROUS_KEYS = {
    "api_key",
    "api_secret",
    "api_token",
    "authorization",
    "cookie",
    "credentials",
    "live_trading",
    "private_key",
    "secret",
    "signature",
    "wallet",
}
_WRITE_PATH_FRAGMENTS = (
    "allowance",
    "approval",
    "cancel",
    "create-order",
    "order/create",
    "/order",
    "private",
    "signature",
    "trade",
    "wallet",
)
_ALLOWED_ENDPOINTS = {
    "gamma_markets_endpoint": (
        {"https"},
        {"gamma-api.polymarket.com"},
        {"/markets"},
    ),
    "resolution_endpoint": (
        {"https"},
        {"gamma-api.polymarket.com"},
        {"/markets"},
    ),
    "binance_depth_endpoint": (
        {"https"},
        {"api.binance.com", "api1.binance.com", "api2.binance.com", "api3.binance.com"},
        {"/api/v3/depth"},
    ),
    "binance_ws_url": (
        {"wss"},
        {"stream.binance.com"},
        {"/ws", "/stream"},
    ),
    "polymarket_ws_url": (
        {"wss"},
        {"ws-subscriptions-clob.polymarket.com"},
        {"/ws/market"},
    ),
    "chainlink_ws_url": (
        {"wss"},
        {"ws-live-data.polymarket.com"},
        {"", "/"},
    ),
}


@dataclass(frozen=True, slots=True)
class OperatorConfig:
    """Complete configuration identity for one paper-only operator."""

    operator_id: str
    strategy_id: str
    paper_account_id: str
    source_commit: str
    output_dir: Path | str
    underlying: str = "BTC"
    market_type: str = "binary_up_down"
    window_duration_ms: int = 900_000
    slug_pattern: str | None = None
    title_pattern: str | None = None
    max_preopen_ms: int = 1_800_000
    gamma_markets_endpoint: str = DEFAULT_GAMMA_MARKETS_ENDPOINT
    resolution_endpoint: str = DEFAULT_GAMMA_MARKETS_ENDPOINT
    binance_depth_endpoint: str = DEFAULT_BINANCE_DEPTH_ENDPOINT
    binance_ws_url: str = DEFAULT_BINANCE_WS_URL
    binance_symbol: str = "BTCUSDT"
    polymarket_ws_url: str = DEFAULT_POLYMARKET_WS_URL
    chainlink_ws_url: str = DEFAULT_CHAINLINK_WS_URL
    chainlink_symbol: str = "btc/usd"
    max_alpha_age_ms: int = 2_000
    max_market_age_ms: int = 5_000
    max_pricing_age_ms: int = 5_000
    reconnect_min_seconds: float = 1.0
    reconnect_max_seconds: float = 30.0
    heartbeat_interval_seconds: float = 10.0
    binance_queue_size: int = 10_000
    market_queue_size: int = 10_000
    binance_delta_buffer_size: int = 5_000
    binance_book_level_limit: int = 5_000
    ofi_ema_alpha: float = 0.2
    ofi_window_ms: int = 60_000
    ofi_min_samples: int = 20
    ofi_clip: float = 3.0
    ofi_max_events: int = 100_000
    pricing_sample_buffer_size: int = 10_000
    twap_window_ms: int = 900_000
    volatility_return_interval_ms: int = 1_000
    volatility_window_ms: int = 300_000
    volatility_min_samples: int = 20
    volatility_max_abs_log_return: float = 0.20
    annualization_seconds: int = 31_536_000
    pricing_ofi_gamma: float = 0.0015
    pricing_min_edge_5m: float = 0.08
    pricing_min_edge_15m: float = 0.05
    pricing_kelly_fraction: float = 0.25
    pricing_tail_cutoff_ms: int = 30_000
    initial_bankroll: float = 1_000.0
    fee_bps: float = 0.0
    max_single_trade_pct: float = 0.05
    max_position_pct: float = 0.25
    max_window_exposure_pct: float = 0.25
    min_order_usd: float = 1.0
    max_spread_allowed: float = 0.08
    slippage_tolerance: float = 0.01
    oms_signal_cache_size: int = 10_000
    snapshot_lru_size: int = 10_000
    execution_history_limit: int = 10_000
    recent_query_default: int = 50
    recent_query_max: int = 500
    status_filename: str = "operator_status.json"
    status_interval_ms: int = 1_000
    logging_level: str = "INFO"
    mock: bool = True
    dry_run: bool = True
    config_check_only: bool = False
    fsync: bool = False
    paper_only: bool = True
    capital_at_risk: bool = False
    broker_exchange_write_enabled: bool = False
    live_exchange_write_enabled: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.output_dir, Path):
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        for name in ("operator_id", "strategy_id", "paper_account_id", "source_commit"):
            _require_text(name, getattr(self, name))
        if Path(self.operator_id).name != self.operator_id:
            raise ValueError("operator_id must be a safe path component")
        if self.underlying != self.underlying.upper() or not self.underlying:
            raise ValueError("underlying must be a non-empty uppercase symbol")
        if not re.fullmatch(r"[A-Z0-9]{5,20}", self.binance_symbol):
            raise ValueError("binance_symbol must be an uppercase exchange symbol")
        if not self.chainlink_symbol.strip():
            raise ValueError("chainlink_symbol must be non-empty")
        for pattern_name in ("slug_pattern", "title_pattern"):
            pattern = getattr(self, pattern_name)
            if pattern is not None:
                try:
                    re.compile(pattern)
                except re.error as exc:
                    raise ValueError(f"{pattern_name} must be a valid regular expression") from exc
        _validate_positive_ints(self)
        _validate_floats(self)
        if self.reconnect_max_seconds < self.reconnect_min_seconds:
            raise ValueError("reconnect bounds are invalid")
        if self.max_position_pct > self.max_window_exposure_pct:
            raise ValueError("risk position cap cannot exceed window exposure cap")
        if self.window_duration_ms not in {300_000, 900_000}:
            raise ValueError("paper operator supports only 5m and 15m windows")
        sources = {"BTC": ("BTCUSDT", "btc/usd"), "ETH": ("ETHUSDT", "eth/usd")}
        if sources.get(self.underlying) != (self.binance_symbol, self.chainlink_symbol):
            raise ValueError("underlying, binance_symbol and chainlink_symbol must match a supported asset")
        canonical_slug = (
            rf"{self.underlying.lower()}-updown-{self.window_duration_ms // 60_000}m-\d+"
        )
        if self.slug_pattern is None:
            object.__setattr__(self, "slug_pattern", canonical_slug)
        elif self.slug_pattern != canonical_slug:
            raise ValueError("slug_pattern must match the configured asset and window duration")
        if self.binance_book_level_limit < 1_000:
            raise ValueError("binance_book_level_limit must accommodate the 1000-level REST bootstrap")
        if self.recent_query_default > self.recent_query_max:
            raise ValueError("recent query default cannot exceed its hard maximum")
        if self.logging_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("logging_level is invalid")
        if Path(self.status_filename).name != self.status_filename:
            raise ValueError("status_filename must be a plain filename")
        if self.status_filename in {"account_checkpoint.json", ".operator.lock"}:
            raise ValueError("status_filename cannot overwrite account checkpoint or ownership lock")
        _validate_safety(self)
        for field_name in _ALLOWED_ENDPOINTS:
            _validate_readonly_endpoint(field_name, str(getattr(self, field_name)))

    @property
    def config_sha256(self) -> str:
        encoded = json.dumps(
            self.config_identity(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def config_identity(self) -> dict[str, object]:
        """Return every effective field in deterministic JSON-native form."""

        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return payload


def load_operator_config(path: str | Path) -> OperatorConfig:
    """Load one strict TOML document without environment-variable overlays."""

    config_path = Path(path)
    with config_path.open("rb") as handle:
        payload = tomllib.load(handle)
    return operator_config_from_mapping(payload)


def operator_config_from_mapping(payload: object) -> OperatorConfig:
    """Validate unknown/dangerous keys before constructing the frozen config."""

    if not isinstance(payload, dict):
        raise ValueError("operator config must be a mapping")
    _reject_dangerous_keys(payload)
    allowed = {field.name for field in fields(OperatorConfig)}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError("unknown operator config fields: " + ", ".join(sorted(unknown)))
    try:
        return OperatorConfig(**payload)
    except TypeError as exc:
        raise ValueError("operator config fields are incomplete or invalid") from exc


def _reject_dangerous_keys(payload: dict[str, Any], *, prefix: str = "") -> None:
    for raw_key, value in payload.items():
        key = str(raw_key).strip().lower()
        path = f"{prefix}.{key}" if prefix else key
        if key in _DANGEROUS_KEYS or any(fragment in key for fragment in ("secret", "token")):
            raise ValueError(f"dangerous operator config field rejected: {path}")
        if isinstance(value, dict):
            _reject_dangerous_keys(value, prefix=path)


def _validate_safety(config: OperatorConfig) -> None:
    if config.paper_only is not True:
        raise ValueError("paper safety requires paper_only=true")
    for name in (
        "capital_at_risk",
        "broker_exchange_write_enabled",
        "live_exchange_write_enabled",
        "polymarket_write_enabled",
        "wallet_signing_enabled",
    ):
        if getattr(config, name) is not False:
            raise ValueError(f"paper safety requires {name}=false")


def _validate_positive_ints(config: OperatorConfig) -> None:
    names = (
        "window_duration_ms",
        "max_preopen_ms",
        "max_alpha_age_ms",
        "max_market_age_ms",
        "max_pricing_age_ms",
        "binance_queue_size",
        "market_queue_size",
        "binance_delta_buffer_size",
        "binance_book_level_limit",
        "ofi_window_ms",
        "ofi_min_samples",
        "ofi_max_events",
        "pricing_sample_buffer_size",
        "twap_window_ms",
        "volatility_return_interval_ms",
        "volatility_window_ms",
        "volatility_min_samples",
        "annualization_seconds",
        "pricing_tail_cutoff_ms",
        "snapshot_lru_size",
        "execution_history_limit",
        "oms_signal_cache_size",
        "recent_query_default",
        "recent_query_max",
        "status_interval_ms",
    )
    for name in names:
        value = getattr(config, name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            label = "queue_size" if "queue_size" in name else name
            raise ValueError(f"{label} must be a positive integer")


def _validate_floats(config: OperatorConfig) -> None:
    positive = ("reconnect_min_seconds", "reconnect_max_seconds", "heartbeat_interval_seconds")
    fractions = (
        "ofi_ema_alpha",
        "max_single_trade_pct",
        "max_position_pct",
        "max_window_exposure_pct",
    )
    non_negative = (
        "fee_bps",
        "max_spread_allowed",
        "slippage_tolerance",
        "pricing_ofi_gamma",
        "pricing_min_edge_5m",
        "pricing_min_edge_15m",
        "pricing_kelly_fraction",
    )
    for name in positive:
        value = float(getattr(config, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    for name in fractions:
        value = float(getattr(config, name))
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError(f"risk/{name} must be in (0, 1]")
    for name in non_negative:
        value = float(getattr(config, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be non-negative and finite")
    if float(config.ofi_clip) <= 0.0 or not math.isfinite(float(config.ofi_clip)):
        raise ValueError("ofi_clip must be positive and finite")
    if not 0.0 < float(config.volatility_max_abs_log_return) <= 1.0:
        raise ValueError("volatility_max_abs_log_return must be in (0, 1]")
    if not math.isfinite(float(config.initial_bankroll)) or config.initial_bankroll <= 0.0:
        raise ValueError("initial_bankroll must be positive and finite")
    if not math.isfinite(float(config.min_order_usd)) or config.min_order_usd <= 0.0:
        raise ValueError("min_order_usd must be positive and finite")
    if not 0.0 <= config.fee_bps <= 10_000.0:
        raise ValueError("fee_bps must be in [0, 10,000]")
    for name in ("pricing_min_edge_5m", "pricing_min_edge_15m", "pricing_kelly_fraction"):
        if float(getattr(config, name)) > 1.0:
            raise ValueError(f"{name} must be in [0, 1]")


def _validate_readonly_endpoint(field_name: str, endpoint: str) -> None:
    parsed = urlsplit(endpoint)
    schemes, hosts, paths = _ALLOWED_ENDPOINTS[field_name]
    normalized_path = parsed.path.rstrip("/") or "/"
    if (
        parsed.scheme not in schemes
        or (parsed.hostname or "").lower() not in hosts
        or normalized_path not in paths
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or any(fragment in normalized_path.lower() for fragment in _WRITE_PATH_FRAGMENTS)
    ):
        raise ValueError(f"{field_name} must be an approved public read-only endpoint")


def _require_text(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
