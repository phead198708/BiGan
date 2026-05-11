"""Realtime ingestion service for Polymarket CLOB market data.

Connects to ``wss://ws-subscriptions-clob.polymarket.com/ws/market``, dynamically
tracks active 15-minute BTC up/down markets via the Gamma API, validates
order-book consistency via the server-supplied ``hash`` field, and persists
every raw message to NDJSON (with an hourly Parquet rollup worker).

Modules:
    config: Settings loaded from env vars / .env.
    message_types: Pydantic models for CLOB ``market`` channel events.
    gamma_client: Async HTTP client over the Polymarket Gamma API.
    clob_ws: Async CLOB WebSocket client (connect, sub/unsub, parse).
    book_state: Local order book replica + hash verification.
    sink: Sink protocol + NDJSON gzip implementation.
    rollup: NDJSON -> Parquet hourly worker.
    metrics: Prometheus metrics registry.
    runner: Orchestrates poll + ws + sink + rollup as long-running service.

Rust hot-path placeholder:
    Message parsing (``message_types``) and hash validation (``book_state``)
    are the two CPU hot spots. Both are factored as pure functions consuming
    bytes / dicts and producing dataclasses, so they can be re-implemented in
    Rust via PyO3 without changing call sites.
"""
