"""Read-only Polymarket RTDS Chainlink BTC/USD collection."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import Counter, deque
from typing import Any, Protocol

import websockets

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.corpus.contracts import safety_fields

DEFAULT_POLYMARKET_RTDS_URL = "wss://ws-live-data.polymarket.com"
POLYMARKET_RTDS_CHAINLINK_TOPIC = "crypto_prices_chainlink"
POLYMARKET_RTDS_CHAINLINK_SYMBOL = "btc/usd"
CHAINLINK_RTDS_RAW_ROW_SCHEMA_VERSION = (
    "bigan-v8-polymarket-chainlink-rtds-raw-price-v1"
)
CHAINLINK_RTDS_RAW_FILENAME = "raw_polymarket_chainlink_prices.jsonl"
CHAINLINK_RTDS_COLLECTION_REPORT_FILENAME = (
    "polymarket_chainlink_rtds_collection_report.json"
)
CHAINLINK_RTDS_CORPUS_FILENAME = "polymarket_chainlink_prices.jsonl"
CHAINLINK_RTDS_CORPUS_MANIFEST_FILENAME = (
    "polymarket_chainlink_decision_time_evidence_manifest.json"
)


class ChainlinkRTDSSnapshotSource(Protocol):
    """Read-only snapshot boundary used by round-scoped collectors."""

    def rows(self) -> list[dict[str, Any]]: ...

    def collection_report(self) -> dict[str, Any]: ...


class ChainlinkRTDSMessageError(ValueError):
    """Raised when an RTDS message cannot enter causal raw evidence."""

    def __init__(self, message: str, *, reason_code: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


def parse_chainlink_rtds_message(
    raw_message: str | bytes,
    *,
    received_at_ts: int,
) -> list[dict[str, Any]]:
    """Normalize one Chainlink subscription snapshot or update message."""

    if isinstance(raw_message, bytes):
        raw_message = raw_message.decode("utf-8")
    if not raw_message.strip():
        return []
    if raw_message.strip().upper() in {"PING", "PONG"}:
        return []
    try:
        message = json.loads(raw_message)
    except json.JSONDecodeError as exc:
        raise ChainlinkRTDSMessageError(
            "RTDS message is not valid JSON",
            reason_code="chainlink_rtds_invalid_json",
        ) from exc
    if not isinstance(message, dict):
        raise ChainlinkRTDSMessageError(
            "RTDS message must be an object",
            reason_code="chainlink_rtds_invalid_message_shape",
        )
    payload = message.get("payload")
    if not isinstance(payload, dict):
        return []
    symbol = str(payload.get("symbol") or "").lower()
    if symbol != POLYMARKET_RTDS_CHAINLINK_SYMBOL:
        return []
    topic = str(message.get("topic") or "")
    message_type = str(message.get("type") or "")
    if topic == POLYMARKET_RTDS_CHAINLINK_TOPIC and message_type == "update":
        values = [payload]
        source_message_type = "chainlink_update"
    elif topic == "crypto_prices" and message_type == "subscribe":
        values = payload.get("data") if isinstance(payload.get("data"), list) else []
        source_message_type = "chainlink_subscription_snapshot"
    else:
        return []

    rows: list[dict[str, Any]] = []
    raw_message_sha256 = canonical_json_sha256(message)
    provider_published_at_ts = _positive_int(message.get("timestamp"))
    for value_row in values:
        if not isinstance(value_row, dict):
            continue
        source_ts = _positive_int(value_row.get("timestamp"))
        price = _positive_float(value_row.get("value"))
        if source_ts is None:
            raise ChainlinkRTDSMessageError(
                "Chainlink RTDS row has no positive source timestamp",
                reason_code="chainlink_rtds_source_timestamp_missing",
            )
        if price is None:
            raise ChainlinkRTDSMessageError(
                "Chainlink RTDS row has no positive price",
                reason_code="chainlink_rtds_price_missing_or_non_positive",
            )
        available_at_ts = max(
            source_ts,
            received_at_ts,
            provider_published_at_ts or 0,
        )
        row = {
            "schema_version": CHAINLINK_RTDS_RAW_ROW_SCHEMA_VERSION,
            "source_type": "polymarket_rtds_chainlink",
            "source_topic": POLYMARKET_RTDS_CHAINLINK_TOPIC,
            "source_message_type": source_message_type,
            "symbol": POLYMARKET_RTDS_CHAINLINK_SYMBOL,
            "source_ts": source_ts,
            "provider_published_at_ts": provider_published_at_ts,
            "received_at_ts": received_at_ts,
            "available_at_ts": available_at_ts,
            "price": price,
            "full_accuracy_value": value_row.get("full_accuracy_value"),
            "raw_message_sha256": raw_message_sha256,
            "timestamp_causality_valid": source_ts <= available_at_ts,
            "read_only": True,
            **safety_fields(),
        }
        row["raw_chainlink_price_row_sha256"] = canonical_json_sha256(row)
        rows.append(row)
    return rows


class PolymarketChainlinkRTDSCollector:
    """Bounded-memory background collector for the public Chainlink RTDS stream."""

    read_only = True
    write_capable = False
    paper_only = True
    capital_at_risk = False
    broker_exchange_write_enabled = False
    live_exchange_write_enabled = False
    polymarket_write_enabled = False
    wallet_signing_enabled = False

    def __init__(
        self,
        *,
        url: str = DEFAULT_POLYMARKET_RTDS_URL,
        max_rows: int = 14_400,
        open_timeout_seconds: float = 10.0,
        receive_poll_seconds: float = 1.0,
        ping_interval_seconds: float = 5.0,
        reconnect_delay_seconds: float = 1.0,
    ) -> None:
        if max_rows <= 0:
            raise ValueError("max_rows must be positive")
        if min(
            open_timeout_seconds,
            receive_poll_seconds,
            ping_interval_seconds,
            reconnect_delay_seconds,
        ) <= 0.0:
            raise ValueError("RTDS timeout and interval values must be positive")
        self.url = url
        self.max_rows = max_rows
        self.open_timeout_seconds = open_timeout_seconds
        self.receive_poll_seconds = receive_poll_seconds
        self.ping_interval_seconds = ping_interval_seconds
        self.reconnect_delay_seconds = reconnect_delay_seconds
        self._rows: deque[dict[str, Any]] = deque(maxlen=max_rows)
        self._row_keys: set[tuple[int, float]] = set()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at_ts: int | None = None
        self._stopped_at_ts: int | None = None
        self._connection_count = 0
        self._reconnect_count = 0
        self._message_count = 0
        self._invalid_reason_counter: Counter[str] = Counter()
        self._last_error_type: str | None = None
        self._last_error_message: str | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._started_at_ts = _now_ms()
        self._stopped_at_ts = None
        self._thread = threading.Thread(
            target=self._run_thread,
            name="v8-polymarket-chainlink-rtds",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, timeout_seconds: float = 5.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(0.001, timeout_seconds))
        self._stopped_at_ts = _now_ms()

    def rows(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self._rows]

    def wait_for_rows(self, *, timeout_seconds: float = 5.0) -> bool:
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        while time.monotonic() < deadline:
            if self.rows():
                return True
            if self._stop_event.is_set():
                return False
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        return bool(self.rows())

    def collection_report(self) -> dict[str, Any]:
        rows = self.rows()
        thread_alive = self._thread is not None and self._thread.is_alive()
        return {
            "report_type": "polymarket_chainlink_rtds_collection",
            "source_type": "polymarket_rtds_chainlink",
            "endpoint": self.url,
            "topic": POLYMARKET_RTDS_CHAINLINK_TOPIC,
            "symbol": POLYMARKET_RTDS_CHAINLINK_SYMBOL,
            "started_at_ts": self._started_at_ts,
            "stopped_at_ts": self._stopped_at_ts,
            "collector_thread_alive": thread_alive,
            "connection_count": self._connection_count,
            "reconnect_count": self._reconnect_count,
            "message_count": self._message_count,
            "raw_price_row_count": len(rows),
            "min_source_ts": min((int(row["source_ts"]) for row in rows), default=None),
            "max_source_ts": max((int(row["source_ts"]) for row in rows), default=None),
            "max_available_at_ts": max(
                (int(row["available_at_ts"]) for row in rows), default=None
            ),
            "timestamp_causality_violation_count": sum(
                1 for row in rows if row.get("timestamp_causality_valid") is not True
            ),
            "invalid_reason_distribution": dict(
                sorted(self._invalid_reason_counter.items())
            ),
            "last_error_type": self._last_error_type,
            "last_error_message": self._last_error_message,
            "decision_critical": False,
            "fail_closed_when_feature_unavailable": True,
            "read_only": True,
            **safety_fields(),
        }

    def _run_thread(self) -> None:
        try:
            asyncio.run(self._run())
        except Exception as exc:  # pragma: no cover - defensive thread boundary
            self._record_error(exc)

    async def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                async with websockets.connect(
                    self.url,
                    open_timeout=self.open_timeout_seconds,
                    close_timeout=1.0,
                    ping_interval=None,
                    max_size=2**22,
                ) as ws:
                    self._connection_count += 1
                    await ws.send(json.dumps(_subscription_payload(), separators=(",", ":")))
                    next_ping = time.monotonic() + self.ping_interval_seconds
                    while not self._stop_event.is_set():
                        try:
                            raw = await asyncio.wait_for(
                                ws.recv(), timeout=self.receive_poll_seconds
                            )
                        except TimeoutError:
                            raw = None
                        if raw is not None:
                            self._message_count += 1
                            self._accept_message(raw)
                        if time.monotonic() >= next_ping:
                            await ws.send("PING")
                            next_ping = time.monotonic() + self.ping_interval_seconds
            except Exception as exc:  # noqa: BLE001
                self._record_error(exc)
                self._reconnect_count += 1
                await _sleep_until_stopped(
                    self._stop_event, self.reconnect_delay_seconds
                )

    def _accept_message(self, raw: str | bytes) -> None:
        try:
            rows = parse_chainlink_rtds_message(raw, received_at_ts=_now_ms())
        except ChainlinkRTDSMessageError as exc:
            self._invalid_reason_counter[exc.reason_code] += 1
            self._record_error(exc)
            return
        with self._lock:
            for row in rows:
                key = (int(row["source_ts"]), float(row["price"]))
                if key in self._row_keys:
                    continue
                if len(self._rows) == self.max_rows:
                    expired = self._rows[0]
                    self._row_keys.discard(
                        (int(expired["source_ts"]), float(expired["price"]))
                    )
                self._rows.append(row)
                self._row_keys.add(key)

    def _record_error(self, exc: Exception) -> None:
        self._last_error_type = exc.__class__.__name__
        self._last_error_message = str(exc)


def _subscription_payload() -> dict[str, Any]:
    return {
        "action": "subscribe",
        "subscriptions": [
            {
                "topic": POLYMARKET_RTDS_CHAINLINK_TOPIC,
                "type": "*",
                "filters": json.dumps(
                    {"symbol": POLYMARKET_RTDS_CHAINLINK_SYMBOL},
                    separators=(",", ":"),
                ),
            }
        ],
    }


async def _sleep_until_stopped(stop_event: threading.Event, seconds: float) -> None:
    await asyncio.to_thread(stop_event.wait, seconds)


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0.0 else None


def _now_ms() -> int:
    return time.time_ns() // 1_000_000
