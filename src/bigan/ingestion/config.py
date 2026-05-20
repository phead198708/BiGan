"""Ingestion service configuration.

Loaded from environment variables (or a local ``.env`` file). All settings are
prefixed with ``BIGAN_`` to avoid collision with other tooling.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class IngestionSettings(BaseSettings):
    """Runtime configuration for the ingestion service."""

    model_config = SettingsConfigDict(
        env_prefix="BIGAN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Endpoints ---
    clob_ws_url: str = Field(
        default="wss://ws-subscriptions-clob.polymarket.com/ws/market",
        description="Polymarket CLOB market channel WebSocket endpoint.",
    )
    coinbase_ws_url: str = Field(
        default="wss://advanced-trade-ws.coinbase.com",
        description="Coinbase Advanced Trade public market-data WebSocket.",
    )
    kraken_ws_url: str = Field(
        default="wss://ws.kraken.com/v2",
        description="Kraken WebSocket v2 endpoint.",
    )
    chainlink_rpc_url: str = Field(
        default="",
        description="Ethereum JSON-RPC URL for Chainlink BTC/USD latestRoundData.",
    )
    chainlink_feed_address: str = Field(
        default="0xF4030086522a5bEEa4988F8cA5B36dbC97BeE88c",
        description="Chainlink BTC/USD AggregatorV3 proxy address.",
    )
    gamma_api_base: str = Field(
        default="https://gamma-api.polymarket.com",
        description="Polymarket Gamma REST API base URL.",
    )

    # --- Market selection ---
    market_slug_prefix: str = Field(
        default="btc-updown-15m-",
        description="Only markets whose slug starts with this prefix will be subscribed.",
    )
    coinbase_product_id: str = Field(default="BTC-USD")
    kraken_symbol: str = Field(default="BTC/USD")
    chainlink_symbol: str = Field(default="BTC/USD")
    gamma_poll_interval_seconds: float = Field(
        default=60.0,
        ge=5.0,
        description="How often to poll Gamma for the active market set.",
    )

    # --- WebSocket reconnect ---
    ws_reconnect_min_seconds: float = Field(default=1.0, ge=0.1)
    ws_reconnect_max_seconds: float = Field(default=30.0, ge=1.0)
    ws_reconnect_reset_after_seconds: float = Field(
        default=60.0,
        ge=0.0,
        description=(
            "Reset exponential reconnect backoff after a connection has stayed "
            "up this long. This prevents isolated remote closes hours apart "
            "from accumulating stale backoff."
        ),
    )
    ws_ping_interval_seconds: float | None = Field(
        default=20.0,
        ge=1.0,
        description=(
            "Optional CLOB client protocol-ping interval. Defaults to enabled "
            "to keep the upstream connection warm while ping-timeout reconnects "
            "are disabled."
        ),
    )
    ws_ping_timeout_seconds: float | None = Field(
        default=None,
        ge=1.0,
        description=(
            "Optional timeout for the CLOB client protocol-ping pong waiter. "
            "Defaults to disabled so a delayed or missing pong does not close "
            "a connection that is still receiving market frames."
        ),
    )
    ws_idle_probe_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        description=(
            "Timeout for the explicit receive-loop idle ping probe that runs only "
            "after ws_message_timeout_seconds elapses without a frame."
        ),
    )
    ws_message_timeout_seconds: float = Field(
        default=45.0,
        ge=5.0,
        description=(
            "If no WebSocket frame is received for this long, send a ping "
            "probe before reconnecting. Keep this below the soak liveness "
            "threshold so a healthy idle ping refreshes the metric first."
        ),
    )
    price_reader_reconnect_min_seconds: float = Field(default=1.0, ge=0.1)
    price_reader_reconnect_max_seconds: float = Field(default=30.0, ge=1.0)
    chainlink_poll_interval_seconds: float = Field(default=5.0, ge=1.0)
    chainlink_request_timeout_seconds: float = Field(default=10.0, ge=1.0)

    # --- Subscription payload tuning ---
    ws_custom_feature_enabled: bool = Field(
        default=True,
        description="When True, enables the best_bid_ask event stream.",
    )

    # --- Storage ---
    data_dir: Path = Field(default=Path("data"))
    raw_subdir: str = Field(default="raw/ws_market")
    rollup_subdir: str = Field(default="rollup/ws_market")
    warehouse_subdir: str = Field(
        default="warehouse",
        description="Root for canonical Parquet tables (issue #3).",
    )
    sink_flush_interval_seconds: float = Field(default=2.0, ge=0.1)
    sink_max_buffer_records: int = Field(default=1000, ge=1)

    # --- Rollup ---
    rollup_enabled: bool = Field(default=True)
    rollup_interval_seconds: float = Field(default=3600.0, ge=60.0)
    rollup_lag_seconds: float = Field(
        default=300.0,
        ge=0.0,
        description="Only roll up files older than this many seconds (to avoid racing the sink).",
    )

    # --- Observability ---
    metrics_port: int = Field(default=9101, ge=1, le=65535)
    metrics_enabled: bool = Field(default=True)
    ingest_lag_warn_seconds: float = Field(
        default=0.5,
        ge=0.0,
        description="Warn when receive_time - message timestamp exceeds this SLA.",
    )
    log_level: str = Field(default="INFO")

    # --- Canonical timestamp contract (issue #23) ---
    timestamp_future_grace_seconds: float = Field(
        default=5.0,
        ge=0.0,
        description="Allow upstream event timestamps to lead ingest time by this many seconds.",
    )
    timestamp_stale_threshold_seconds: float = Field(
        default=600.0,
        ge=0.0,
        description="Quarantine rows whose ingest time lags event time by more than this many seconds.",
    )

    # --- Resilience ---
    book_hash_mismatch_max_retries: int = Field(
        default=3,
        ge=0,
        description="On hash mismatch we resubscribe to receive a fresh snapshot; retry budget per minute.",
    )

    # --- Gap detection / REST backfill (issue #5) ---
    clob_rest_url: str = Field(
        default="https://clob.polymarket.com",
        description="Polymarket CLOB REST API base URL for orderbook reads.",
    )
    polymarket_data_api_url: str = Field(
        default="https://data-api.polymarket.com",
        description="Polymarket public Data API base URL for trade-history reads.",
    )
    gap_detection_enabled: bool = Field(
        default=True,
        description="Enable per-asset silence detection and REST backfill on resume.",
    )
    gap_silence_threshold_seconds: float = Field(
        default=30.0,
        ge=1.0,
        description="Asset silent for this long is considered to be in a gap.",
    )
    gap_min_resume_seconds: float = Field(
        default=1.0,
        ge=0.0,
        description="Minimum delta from gap_start before a resume is honoured.",
    )
    gap_check_interval_seconds: float = Field(
        default=5.0,
        ge=0.5,
        description="How often the watchdog scans for newly-silent assets.",
    )
    backfill_rest_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
    )
    backfill_max_pages: int = Field(
        default=20,
        ge=1,
        description="Max trade history pages to scan per gap (safety cap).",
    )
    backfill_max_concurrency: int = Field(
        default=4,
        ge=1,
        description="Max concurrent REST backfill invocations.",
    )
    backfill_rate_limit_per_second: float = Field(
        default=10.0,
        gt=0.0,
        description="Global CLOB REST request rate limit for backfill.",
    )
    backfill_circuit_failure_threshold: int = Field(
        default=5,
        ge=1,
        description="Consecutive REST failures before opening the backfill circuit.",
    )
    backfill_circuit_cool_down_seconds: float = Field(
        default=30.0,
        ge=0.0,
        description="How long the open backfill circuit waits before half-open probe.",
    )
    initial_snapshot_enabled: bool = Field(
        default=True,
        description=(
            "Fetch an immediate CLOB REST book snapshot for each Gamma-discovered "
            "asset instead of waiting for the WebSocket to emit its first book."
        ),
    )

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / self.raw_subdir

    @property
    def rollup_dir(self) -> Path:
        return self.data_dir / self.rollup_subdir

    @property
    def warehouse_dir(self) -> Path:
        return self.data_dir / self.warehouse_subdir
