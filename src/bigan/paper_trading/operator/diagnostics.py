"""Bounded, allowlisted diagnostics; never retain payloads or exception text."""

from __future__ import annotations

from collections import deque
from enum import StrEnum


class DiagnosticCode(StrEnum):
    EVENT_FROM_FUTURE = "EVENT_FROM_FUTURE"
    EVENT_OUT_OF_ORDER = "EVENT_OUT_OF_ORDER"
    SYMBOL_MISMATCH = "SYMBOL_MISMATCH"
    DEPTH_INVALID = "DEPTH_INVALID"
    DEPTH_EMPTY_SIDE = "DEPTH_EMPTY_SIDE"
    DEPTH_CROSSED = "DEPTH_CROSSED"
    DEPTH_LEVEL_LIMIT = "DEPTH_LEVEL_LIMIT"
    DEPTH_INVALID_PRICE = "DEPTH_INVALID_PRICE"
    DEPTH_INVALID_SIZE = "DEPTH_INVALID_SIZE"
    DEPTH_DUPLICATE_PRICE = "DEPTH_DUPLICATE_PRICE"
    DEPTH_MISSING_FULL_BOOK = "DEPTH_MISSING_FULL_BOOK"
    DEPTH_SEQUENCE_GAP = "DEPTH_SEQUENCE_GAP"
    DEPTH_BUFFER_OVERFLOW = "DEPTH_BUFFER_OVERFLOW"
    DEPTH_TOP_MISMATCH = "DEPTH_TOP_MISMATCH"
    DEPTH_TOP_TIMEOUT = "DEPTH_TOP_TIMEOUT"
    DEPTH_MISSING_TOP = "DEPTH_MISSING_TOP"
    DEPTH_PARSER_REJECTED = "DEPTH_PARSER_REJECTED"
    PRICING_MISSING_SAMPLES = "PRICING_MISSING_SAMPLES"
    PRICING_FUTURE_SPOT = "PRICING_FUTURE_SPOT"
    PRICING_FUTURE_ORACLE = "PRICING_FUTURE_ORACLE"
    PRICING_STALE_SPOT = "PRICING_STALE_SPOT"
    PRICING_STALE_ORACLE = "PRICING_STALE_ORACLE"
    PRICING_VOLATILITY_WARMUP = "PRICING_VOLATILITY_WARMUP"
    PRICING_TWAP_UNAVAILABLE = "PRICING_TWAP_UNAVAILABLE"
    WS_HEARTBEAT_TIMEOUT = "WS_HEARTBEAT_TIMEOUT"
    WS_TIMEOUT = "WS_TIMEOUT"
    WS_CLOSED = "WS_CLOSED"
    WS_HTTP_FAILURE = "WS_HTTP_FAILURE"
    WS_INVALID_PAYLOAD = "WS_INVALID_PAYLOAD"
    WS_REBOOTSTRAP_REQUIRED = "WS_REBOOTSTRAP_REQUIRED"
    WS_PROTOCOL_ERROR = "WS_PROTOCOL_ERROR"
    WS_IO_FAILURE = "WS_IO_FAILURE"


NUMERIC_FIELDS = frozenset({
    "timestamp_ms", "event_timestamp_ms", "received_at_ms", "generation",
    "spot_timestamp_ms", "oracle_timestamp_ms", "return_sample_count",
    "expected_update_id", "first_update_id", "last_update_id",
})
MAX_COUNTER = 2**63 - 1


class DiagnosticBuffer:
    """Per-component counters plus the latest 32 events, retained across reconnects."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.recent: deque[dict[str, object]] = deque(maxlen=32)

    def record(self, code: DiagnosticCode, **context: int | None) -> None:
        if not isinstance(code, DiagnosticCode) or set(context) - NUMERIC_FIELDS:
            raise ValueError("diagnostic fields must be allowlisted")
        event: dict[str, object] = {"code": code.value}
        for key, value in context.items():
            if type(value) is int and -MAX_COUNTER <= value <= MAX_COUNTER:
                event[key] = value
        self.counts[code.value] = min(MAX_COUNTER, self.counts.get(code.value, 0) + 1)
        self.recent.append(event)

    def to_dict(self) -> dict[str, object]:
        return {"counts": dict(self.counts), "recent": [dict(item) for item in self.recent]}


class DepthValidationError(ValueError):
    def __init__(self, code: DiagnosticCode, message: str) -> None:
        super().__init__(message)
        self.code = code


class FeedResyncRequired(ConnectionError):
    pass


class HeartbeatTimeout(TimeoutError):
    pass
