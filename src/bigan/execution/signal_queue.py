"""Executor-ready prediction signal JSONL queue helpers."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from bigan.execution.v6_gate import (
    V6JointGateConfig,
    build_v6_signal_fields,
    is_v6_model_version,
)
from bigan.features.low_latency import JsonlRawQueue, RawQueueItem
from bigan.monitoring.events import prediction_event_from_prediction_row
from bigan.monitoring.market_quality import (
    round_end_ts_from_canonical_symbol,
    tradable_market_implied_probability,
)


@dataclass(frozen=True, slots=True)
class ExecutionSignal:
    """One executor-ready signal row for ``--signal-jsonl-path``."""

    event_id: str
    ts: int
    created_at: int
    model_version: str
    prob_up_15m: float
    canonical_symbol: str
    token_id: str
    outcome_side: str
    round_slug: str
    round_end_ts: int
    market_implied_prob: float
    token_probability: float
    edge: float
    bridged_at: int
    opposite_token_id: str = ""
    model_probability: float | None = None
    polymarket_price: float | None = None
    mispricing_edge: float | None = None
    p_up: float | None = None
    p_down: float | None = None
    p_neutral: float | None = None
    p_vol_up: float | None = None
    p_vol_down: float | None = None
    v6_joint_side: str | None = None
    settlement_residual: float | None = None
    token_expected_win_probability: float | None = None
    p_up_residual_adjusted: float | None = None
    p_down_residual_adjusted: float | None = None
    expected_edge_up: float | None = None
    expected_edge_down: float | None = None
    residual_expected_edge_up: float | None = None
    residual_expected_edge_down: float | None = None
    p_up_hit_5c_before_loss_10c: float | None = None
    p_up_hit_10c_before_loss_10c: float | None = None
    p_up_loss_10c_before_hit_5c: float | None = None
    p_down_hit_5c_before_loss_10c: float | None = None
    p_down_hit_10c_before_loss_10c: float | None = None
    p_down_loss_10c_before_hit_5c: float | None = None
    selected_hit_5c_before_loss_10c: float | None = None
    selected_hit_10c_before_loss_10c: float | None = None
    selected_loss_10c_before_hit_5c: float | None = None
    selected_confidence_score: float | None = None
    selected_side: str | None = None
    selected_expected_edge: float | None = None
    entry_worst_price: float | None = None
    should_enter_settlement: bool | None = None


@dataclass(frozen=True, slots=True)
class EventDrivenSignalQueueReport:
    """Summary for one v7 event-driven signal repricing batch."""

    rows_read: int = 0
    top_of_book_rows_applied: int = 0
    base_signals_loaded: int = 0
    signals_written: int = 0
    start_offset: int = 0
    next_offset: int = 0
    current_round_slug: str | None = None
    buckets_seen: int = 0
    buckets_emitted: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SignalBridgeDiagnostics:
    """Explain why prediction rows did or did not become executor signals."""

    rows_read: int = 0
    signals_built: int = 0
    current_round_slug: str | None = None
    current_round_signal_count: int = 0
    fresh_current_round_signal_count: int = 0
    stale_current_round_signal_count: int = 0
    active_signal_count: int = 0
    expired_signal_count: int = 0
    non_current_round_signal_count: int = 0
    reason_counts: dict[str, int] = field(default_factory=dict)
    round_counts: dict[str, int] = field(default_factory=dict)
    latest_event_ts: int | None = None
    latest_created_at: int | None = None
    min_signal_age_seconds: float | None = None
    max_signal_age_seconds: float | None = None
    max_event_age_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SignalCursor:
    """Opaque read position for a signal transport."""

    position: int = 0
    signature: str = ""


@dataclass(frozen=True, slots=True)
class SignalPayloadBatch:
    payloads: list[dict[str, Any]]
    cursor: SignalCursor


class SignalSink(Protocol):
    def append_current_round(
        self,
        signals: list[ExecutionSignal],
        *,
        reference_ms: int,
        max_event_age_seconds: float | None = None,
    ) -> int:
        ...


class SignalSource(Protocol):
    def latest_cursor(self, *, start: str) -> SignalCursor:
        ...

    def read_after(
        self,
        cursor: SignalCursor,
        *,
        limit: int | None = None,
    ) -> SignalPayloadBatch:
        ...


class JsonlSignalSink:
    """JSONL-backed signal sink used by the current paper executor data plane."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def append_current_round(
        self,
        signals: list[ExecutionSignal],
        *,
        reference_ms: int,
        max_event_age_seconds: float | None = None,
    ) -> int:
        if not signals:
            return 0
        current_round_slug = _current_round_slug(signals, now_ms=reference_ms)
        if current_round_slug is None:
            if self.path.exists():
                self.path.write_text("", encoding="utf-8")
            return 0
        round_signals = [
            signal for signal in signals if signal.round_slug == current_round_slug
        ]
        return self.append_round_signals(
            round_signals,
            current_round_slug=current_round_slug,
            max_event_age_seconds=max_event_age_seconds,
            identity_fn=_signal_identity,
            payload_identity_fn=_payload_signal_identity,
        )

    def append_round_signals(
        self,
        signals: list[ExecutionSignal],
        *,
        current_round_slug: str,
        max_event_age_seconds: float | None = None,
        identity_fn: Any,
        payload_identity_fn: Any,
    ) -> int:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existing_round_slug = _existing_queue_round_slug(self.path)
        mode = "w" if existing_round_slug and existing_round_slug != current_round_slug else "a"
        if not signals:
            if mode == "w":
                self.path.write_text("", encoding="utf-8")
            return 0
        fresh_signals: list[ExecutionSignal] = []
        for signal in signals:
            if (
                max_event_age_seconds is not None
                and _signal_age_seconds(signal) > max_event_age_seconds
            ):
                continue
            fresh_signals.append(signal)
        if not fresh_signals:
            if mode == "w":
                self.path.write_text("", encoding="utf-8")
            return 0
        existing_identities: set[Any] = set()
        if mode == "a":
            existing_identities = _existing_payload_identities(
                self.path,
                current_round_slug,
                payload_identity_fn=payload_identity_fn,
            )
        rows: list[str] = []
        for signal in fresh_signals:
            identity = identity_fn(signal)
            if identity in existing_identities:
                continue
            existing_identities.add(identity)
            rows.append(json.dumps(asdict(signal), sort_keys=True))
        if not rows:
            return 0
        with self.path.open(mode, encoding="utf-8") as handle:
            for row in rows:
                handle.write(row + "\n")
        return len(rows)


class JsonlSignalSource:
    """JSONL-backed signal source with cursor rotation detection."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def latest_cursor(self, *, start: str) -> SignalCursor:
        if not self.path.exists() or start != "tail":
            return SignalCursor()
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return SignalCursor()
        position = len(lines)
        return SignalCursor(
            position=position,
            signature=signal_jsonl_prefix_signature(lines, position),
        )

    def read_after(
        self,
        cursor: SignalCursor,
        *,
        limit: int | None = None,
    ) -> SignalPayloadBatch:
        if not self.path.exists():
            return SignalPayloadBatch(payloads=[], cursor=cursor)
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return SignalPayloadBatch(payloads=[], cursor=cursor)
        after_position = cursor.position
        if after_position > len(lines):
            after_position = 0
        if (
            after_position > 0
            and cursor.signature
            and signal_jsonl_prefix_signature(lines, after_position) != cursor.signature
        ):
            after_position = 0
        payloads: list[dict[str, Any]] = []
        last_position = after_position
        for idx, line in enumerate(lines[after_position:], start=after_position + 1):
            last_position = idx
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            payloads.append(payload)
            if limit is not None and len(payloads) >= limit:
                break
        return SignalPayloadBatch(
            payloads=payloads,
            cursor=SignalCursor(
                position=last_position,
                signature=signal_jsonl_prefix_signature(lines, last_position),
            ),
        )


class KafkaSignalSink:
    """Kafka-backed signal sink.

    The Kafka client is optional at import time so local paper/replay workflows
    can keep using JSONL without installing broker dependencies.
    """

    def __init__(
        self,
        *,
        bootstrap_servers: str,
        topic: str,
        producer: Any | None = None,
        producer_config: Mapping[str, Any] | None = None,
        flush_timeout_seconds: float = 5.0,
    ) -> None:
        if not bootstrap_servers:
            raise ValueError("bootstrap_servers is required")
        if not topic:
            raise ValueError("topic is required")
        if flush_timeout_seconds < 0:
            raise ValueError("flush_timeout_seconds must be non-negative")
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.flush_timeout_seconds = flush_timeout_seconds
        self._producer = producer
        self._producer_config = dict(producer_config or {})

    def append_current_round(
        self,
        signals: list[ExecutionSignal],
        *,
        reference_ms: int,
        max_event_age_seconds: float | None = None,
    ) -> int:
        if not signals:
            return 0
        current_round_slug = _current_round_slug(signals, now_ms=reference_ms)
        if current_round_slug is None:
            return 0
        round_signals = [
            signal for signal in signals if signal.round_slug == current_round_slug
        ]
        return self.append_round_signals(
            round_signals,
            max_event_age_seconds=max_event_age_seconds,
            identity_fn=_signal_identity,
        )

    def append_round_signals(
        self,
        signals: list[ExecutionSignal],
        *,
        max_event_age_seconds: float | None = None,
        identity_fn: Any,
    ) -> int:
        producer = self._ensure_producer()
        seen_identities: set[Any] = set()
        written = 0
        for signal in signals:
            if (
                max_event_age_seconds is not None
                and _signal_age_seconds(signal) > max_event_age_seconds
            ):
                continue
            identity = identity_fn(signal)
            if identity in seen_identities:
                continue
            seen_identities.add(identity)
            producer.produce(
                self.topic,
                key=_kafka_signal_key(signal),
                value=json.dumps(asdict(signal), sort_keys=True).encode("utf-8"),
            )
            written += 1
        if written:
            producer.flush(self.flush_timeout_seconds)
        return written

    def _ensure_producer(self) -> Any:
        if self._producer is not None:
            return self._producer
        kafka = _load_confluent_kafka()
        config = {
            "bootstrap.servers": self.bootstrap_servers,
            **self._producer_config,
        }
        self._producer = kafka.Producer(config)
        return self._producer


class KafkaSignalSource:
    """Kafka-backed signal source for executor consumption."""

    def __init__(
        self,
        *,
        bootstrap_servers: str,
        topic: str,
        group_id: str,
        consumer: Any | None = None,
        consumer_config: Mapping[str, Any] | None = None,
        poll_timeout_seconds: float = 0.25,
        max_records: int = 500,
    ) -> None:
        if not bootstrap_servers:
            raise ValueError("bootstrap_servers is required")
        if not topic:
            raise ValueError("topic is required")
        if not group_id:
            raise ValueError("group_id is required")
        if poll_timeout_seconds < 0:
            raise ValueError("poll_timeout_seconds must be non-negative")
        if max_records <= 0:
            raise ValueError("max_records must be positive")
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.poll_timeout_seconds = poll_timeout_seconds
        self.max_records = max_records
        self._consumer = consumer
        self._consumer_config = dict(consumer_config or {})
        self._consumer_start: str | None = None

    def latest_cursor(self, *, start: str) -> SignalCursor:
        self._ensure_consumer(start=start)
        return SignalCursor()

    def read_after(
        self,
        cursor: SignalCursor,
        *,
        limit: int | None = None,
    ) -> SignalPayloadBatch:
        consumer = self._ensure_consumer(start="tail")
        record_limit = self.max_records if limit is None else min(limit, self.max_records)
        payloads: list[dict[str, Any]] = []
        records_seen = 0
        for _ in range(record_limit):
            message = consumer.poll(self.poll_timeout_seconds)
            if message is None:
                break
            records_seen += 1
            error = message.error() if hasattr(message, "error") else None
            if error:
                continue
            value = message.value()
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            if not value:
                continue
            try:
                payload = json.loads(value)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                payloads.append(payload)
        return SignalPayloadBatch(
            payloads=payloads,
            cursor=SignalCursor(position=cursor.position + records_seen),
        )

    def _ensure_consumer(self, *, start: str) -> Any:
        if self._consumer is not None:
            if self._consumer_start is None:
                self._consumer_start = start
                if hasattr(self._consumer, "subscribe"):
                    self._consumer.subscribe([self.topic])
            return self._consumer
        kafka = _load_confluent_kafka()
        reset = "earliest" if start == "beginning" else "latest"
        config = {
            "bootstrap.servers": self.bootstrap_servers,
            "group.id": self.group_id,
            "auto.offset.reset": reset,
            "enable.auto.commit": True,
            **self._consumer_config,
        }
        self._consumer = kafka.Consumer(config)
        self._consumer.subscribe([self.topic])
        self._consumer_start = start
        return self._consumer


def append_prediction_rows_to_signal_sink(
    sink: SignalSink,
    rows: list[dict[str, Any]],
    *,
    model_version: str,
    allowed_families: frozenset[str] = frozenset({"BTC-15M"}),
    allowed_outcome_sides: frozenset[str] | None = None,
    v6_joint_config: V6JointGateConfig | None = None,
    bridged_at: int | None = None,
    max_event_age_seconds: float | None = None,
    token_ids_by_market_side: Mapping[tuple[str, str, str], str] | None = None,
) -> int:
    if max_event_age_seconds is not None and max_event_age_seconds <= 0:
        raise ValueError("max_event_age_seconds must be positive")
    signals = build_execution_signals_from_prediction_rows(
        rows,
        model_version=model_version,
        allowed_families=allowed_families,
        allowed_outcome_sides=allowed_outcome_sides,
        v6_joint_config=v6_joint_config,
        bridged_at=bridged_at,
        token_ids_by_market_side=token_ids_by_market_side,
    )
    if not signals:
        return 0
    reference_ms = _now_ms() if bridged_at is None else int(bridged_at)
    return sink.append_current_round(
        signals,
        reference_ms=reference_ms,
        max_event_age_seconds=max_event_age_seconds,
    )


def diagnose_prediction_rows_for_signal_bridge(
    rows: list[dict[str, Any]],
    *,
    model_version: str,
    allowed_families: frozenset[str] = frozenset({"BTC-15M"}),
    allowed_outcome_sides: frozenset[str] | None = None,
    v6_joint_config: V6JointGateConfig | None = None,
    bridged_at: int | None = None,
    max_event_age_seconds: float | None = None,
    token_ids_by_market_side: Mapping[tuple[str, str, str], str] | None = None,
) -> SignalBridgeDiagnostics:
    """Return bridge drop reasons without writing to JSONL or Kafka."""

    if max_event_age_seconds is not None and max_event_age_seconds <= 0:
        raise ValueError("max_event_age_seconds must be positive")
    signals = build_execution_signals_from_prediction_rows(
        rows,
        model_version=model_version,
        allowed_families=allowed_families,
        allowed_outcome_sides=allowed_outcome_sides,
        v6_joint_config=v6_joint_config,
        bridged_at=bridged_at,
        token_ids_by_market_side=token_ids_by_market_side,
    )
    reason_counts: dict[str, int] = {}
    if not rows:
        reason_counts["no_prediction_rows"] = 1
    if not signals:
        if rows:
            reason_counts["no_buildable_signals"] = len(rows)
        return SignalBridgeDiagnostics(
            rows_read=len(rows),
            signals_built=0,
            reason_counts=reason_counts,
            max_event_age_seconds=max_event_age_seconds,
        )

    reference_ms = _now_ms() if bridged_at is None else int(bridged_at)
    current_round_slug = _current_round_slug(signals, now_ms=reference_ms)
    round_counts = _round_signal_counts(signals)
    active_signals = [signal for signal in signals if signal.round_end_ts >= reference_ms]
    current_signals = [
        signal for signal in signals if signal.round_slug == current_round_slug
    ]
    if max_event_age_seconds is None:
        fresh_current_signals = list(current_signals)
    else:
        fresh_current_signals = [
            signal
            for signal in current_signals
            if _signal_age_seconds(signal) <= max_event_age_seconds
        ]
    ages = [_signal_age_seconds(signal) for signal in signals]
    if current_round_slug is None:
        reason_counts["no_active_round"] = len(signals)
    elif not current_signals:
        reason_counts["no_current_round_signals"] = len(signals)
    elif not fresh_current_signals:
        reason_counts["current_round_signals_stale"] = len(current_signals)
    else:
        reason_counts["writable_current_round_signals"] = len(fresh_current_signals)
    return SignalBridgeDiagnostics(
        rows_read=len(rows),
        signals_built=len(signals),
        current_round_slug=current_round_slug,
        current_round_signal_count=len(current_signals),
        fresh_current_round_signal_count=len(fresh_current_signals),
        stale_current_round_signal_count=len(current_signals) - len(fresh_current_signals),
        active_signal_count=len(active_signals),
        expired_signal_count=len(signals) - len(active_signals),
        non_current_round_signal_count=len(signals) - len(current_signals),
        reason_counts=reason_counts,
        round_counts=round_counts,
        latest_event_ts=max(signal.ts for signal in signals),
        latest_created_at=max(signal.created_at for signal in signals),
        min_signal_age_seconds=min(ages) if ages else None,
        max_signal_age_seconds=max(ages) if ages else None,
        max_event_age_seconds=max_event_age_seconds,
    )


def append_prediction_rows_as_signal_kafka(
    *,
    bootstrap_servers: str,
    topic: str,
    rows: list[dict[str, Any]],
    model_version: str,
    allowed_families: frozenset[str] = frozenset({"BTC-15M"}),
    allowed_outcome_sides: frozenset[str] | None = None,
    v6_joint_config: V6JointGateConfig | None = None,
    bridged_at: int | None = None,
    max_event_age_seconds: float | None = None,
    token_ids_by_market_side: Mapping[tuple[str, str, str], str] | None = None,
    flush_timeout_seconds: float = 5.0,
) -> int:
    return append_prediction_rows_to_signal_sink(
        KafkaSignalSink(
            bootstrap_servers=bootstrap_servers,
            topic=topic,
            flush_timeout_seconds=flush_timeout_seconds,
        ),
        rows,
        model_version=model_version,
        allowed_families=allowed_families,
        allowed_outcome_sides=allowed_outcome_sides,
        v6_joint_config=v6_joint_config,
        bridged_at=bridged_at,
        max_event_age_seconds=max_event_age_seconds,
        token_ids_by_market_side=token_ids_by_market_side,
    )


def append_prediction_rows_as_signal_jsonl(
    path: Path | str,
    rows: list[dict[str, Any]],
    *,
    model_version: str,
    allowed_families: frozenset[str] = frozenset({"BTC-15M"}),
    allowed_outcome_sides: frozenset[str] | None = None,
    v6_joint_config: V6JointGateConfig | None = None,
    bridged_at: int | None = None,
    max_event_age_seconds: float | None = None,
    token_ids_by_market_side: Mapping[tuple[str, str, str], str] | None = None,
) -> int:
    """Append executor-ready signal rows built directly from prediction rows."""

    return append_prediction_rows_to_signal_sink(
        JsonlSignalSink(path),
        rows,
        model_version=model_version,
        allowed_families=allowed_families,
        allowed_outcome_sides=allowed_outcome_sides,
        v6_joint_config=v6_joint_config,
        bridged_at=bridged_at,
        max_event_age_seconds=max_event_age_seconds,
        token_ids_by_market_side=token_ids_by_market_side,
    )


def append_event_driven_v7_signals_from_raw_queue(
    path: Path | str,
    *,
    base_signal_jsonl_path: Path | str,
    raw_queue_path: Path | str,
    cursor_path: Path | str | None = None,
    bucket_seconds: float = 10.0,
    allowed_families: frozenset[str] = frozenset({"BTC-15M"}),
    max_records: int | None = 20_000,
    bridged_at: int | None = None,
    max_base_signal_age_seconds: float | None = 180.0,
    max_event_age_seconds: float | None = 30.0,
    start: str = "beginning",
) -> EventDrivenSignalQueueReport:
    """Append sub-minute v7 executor signals by repricing base signals.

    The base v7 model signal queue remains the belief source. This helper uses
    new raw top-of-book rows to update the tradable market price and residual
    edge on a fixed 5s/10s bucket cadence.
    """

    if bucket_seconds <= 0:
        raise ValueError("bucket_seconds must be positive")
    if max_records is not None and max_records <= 0:
        raise ValueError("max_records must be positive when set")
    if max_base_signal_age_seconds is not None and max_base_signal_age_seconds <= 0:
        raise ValueError("max_base_signal_age_seconds must be positive when set")
    if max_event_age_seconds is not None and max_event_age_seconds <= 0:
        raise ValueError("max_event_age_seconds must be positive when set")
    if start not in {"beginning", "tail"}:
        raise ValueError("start must be beginning or tail")

    emitted_at = _now_ms() if bridged_at is None else int(bridged_at)
    bucket_ms = int(bucket_seconds * 1000)
    current_bucket_ts = (emitted_at // bucket_ms) * bucket_ms
    base_payloads = _read_signal_payloads(Path(base_signal_jsonl_path))
    current_round_slug = _current_round_slug_from_payloads(base_payloads, now_ms=emitted_at)
    output_path = Path(path)
    if current_round_slug is None:
        if output_path.exists():
            output_path.write_text("", encoding="utf-8")
        return EventDrivenSignalQueueReport(
            base_signals_loaded=len(base_payloads),
            current_round_slug=None,
        )
    base_by_symbol = _latest_v7_base_payloads_by_symbol(
        base_payloads,
        current_round_slug=current_round_slug,
        now_ms=emitted_at,
        max_base_signal_age_seconds=max_base_signal_age_seconds,
        allowed_families=allowed_families,
    )
    start_offset = _read_raw_cursor(Path(cursor_path) if cursor_path is not None else None)
    queue_path = Path(raw_queue_path)
    if cursor_path is not None and start_offset is None and start == "tail" and queue_path.exists():
        start_offset = queue_path.stat().st_size
    if start_offset is None:
        start_offset = 0
    queue = JsonlRawQueue(queue_path)
    items, next_offset, _lines_read = queue.read_from_offset(start_offset, max_records=max_records)
    latest_quotes = _latest_top_of_book_by_bucket(
        items,
        current_round_slug=current_round_slug,
        bucket_seconds=bucket_seconds,
        allowed_families=allowed_families,
        base_by_symbol=base_by_symbol,
    )
    signals_by_bucket: dict[tuple[str, int], ExecutionSignal] = {}
    for (canonical_symbol, bucket_ts), quote in sorted(latest_quotes.items()):
        if bucket_ts >= current_bucket_ts:
            continue
        base = base_by_symbol.get(canonical_symbol)
        if base is None:
            continue
        if bucket_ts <= int(base.get("ts") or 0):
            continue
        signal = _event_driven_v7_signal_from_quote(
            base,
            quote,
            bucket_ts=bucket_ts,
            bridged_at=emitted_at,
        )
        if signal is None:
            continue
        key = (signal.round_slug, signal.ts)
        previous = signals_by_bucket.get(key)
        if previous is None or signal.edge > previous.edge:
            signals_by_bucket[key] = signal
    signals = list(signals_by_bucket.values())
    if max_event_age_seconds is not None:
        signals = [
            signal
            for signal in signals
            if _signal_age_seconds(signal) <= max_event_age_seconds
        ]
    signals_written = _append_event_driven_signals(
        output_path,
        signals,
        current_round_slug=current_round_slug,
    )
    if cursor_path is not None:
        _write_raw_cursor(Path(cursor_path), next_offset)
    return EventDrivenSignalQueueReport(
        rows_read=len(items),
        top_of_book_rows_applied=sum(1 for item in items if item.table == "raw_top_of_book"),
        base_signals_loaded=len(base_payloads),
        signals_written=signals_written,
        start_offset=start_offset,
        next_offset=next_offset,
        current_round_slug=current_round_slug,
        buckets_seen=len(latest_quotes),
        buckets_emitted=len(signals),
    )


def build_execution_signals_from_prediction_rows(
    rows: list[dict[str, Any]],
    *,
    model_version: str,
    allowed_families: frozenset[str] = frozenset({"BTC-15M"}),
    allowed_outcome_sides: frozenset[str] | None = None,
    v6_joint_config: V6JointGateConfig | None = None,
    bridged_at: int | None = None,
    token_ids_by_market_side: Mapping[tuple[str, str, str], str] | None = None,
) -> list[ExecutionSignal]:
    """Build executor-ready signals without rereading ``prediction_events``."""

    token_by_side = dict(token_ids_by_market_side or {})
    token_by_side.update(_token_ids_by_market_side(rows))
    emitted_at = _now_ms() if bridged_at is None else int(bridged_at)
    signals: list[ExecutionSignal] = []
    for row in rows:
        signal = _execution_signal_from_prediction_row(
            row,
            model_version=model_version,
            allowed_families=allowed_families,
            allowed_outcome_sides=allowed_outcome_sides,
            v6_joint_config=v6_joint_config,
            token_by_side=token_by_side,
            bridged_at=emitted_at,
        )
        if signal is not None:
            signals.append(signal)
    return signals


def _execution_signal_from_prediction_row(
    row: dict[str, Any],
    *,
    model_version: str,
    allowed_families: frozenset[str],
    allowed_outcome_sides: frozenset[str] | None,
    v6_joint_config: V6JointGateConfig | None,
    token_by_side: dict[tuple[str, str, str], str],
    bridged_at: int,
) -> ExecutionSignal | None:
    event = prediction_event_from_prediction_row(row)
    snapshot = _snapshot_from_prediction_row(row)
    canonical_symbol = str(snapshot.get("canonical_symbol") or snapshot.get("symbol") or "")
    parsed = _parse_canonical_symbol(canonical_symbol)
    if parsed is None:
        return None
    family, round_slug, token_side = parsed
    if family not in allowed_families or token_side not in {"UP", "DOWN"}:
        return None
    if (
        allowed_outcome_sides is not None
        and v6_joint_config is None
        and token_side not in allowed_outcome_sides
    ):
        return None
    round_end_ts = round_end_ts_from_canonical_symbol(canonical_symbol)
    if round_end_ts is None:
        return None
    opposite_side = "DOWN" if token_side == "UP" else "UP"
    opposite_token_id = token_by_side.get((family, round_slug, opposite_side), "")
    created_at = int(row.get("created_at") or row.get("ingest_ts") or _now_ms())
    if v6_joint_config is not None and is_v6_model_version(model_version):
        fields = build_v6_signal_fields(
            event_id=event.event_id,
            ts=int(event.ts),
            created_at=created_at,
            snapshot=snapshot,
            model_version=model_version,
            config=v6_joint_config,
            round_end_ts=int(round_end_ts),
            bridged_at=bridged_at,
            opposite_token_id=opposite_token_id,
        )
        if fields is None:
            return None
        if allowed_outcome_sides is not None and fields["outcome_side"] not in allowed_outcome_sides:
            return None
        return ExecutionSignal(
            event_id=str(fields["event_id"]),
            ts=int(fields["ts"]),
            created_at=int(fields["created_at"]),
            model_version=model_version,
            prob_up_15m=float(fields["prob_up_15m"]),
            canonical_symbol=str(fields["canonical_symbol"]),
            token_id=str(fields["token_id"]),
            outcome_side=str(fields["outcome_side"]),
            round_slug=str(fields["round_slug"]),
            round_end_ts=int(fields["round_end_ts"]),
            market_implied_prob=float(fields["market_implied_prob"]),
            token_probability=float(fields["token_probability"]),
            edge=float(fields["edge"]),
            bridged_at=int(fields["bridged_at"]),
            opposite_token_id=str(fields.get("opposite_token_id") or ""),
            p_up=float(fields["p_up"]),
            p_down=float(fields["p_down"]),
            p_neutral=float(fields["p_neutral"]),
            p_vol_up=float(fields["p_vol_up"]),
            p_vol_down=float(fields["p_vol_down"]),
            v6_joint_side=(
                str(fields["v6_joint_side"]) if fields.get("v6_joint_side") else None
            ),
        )

    if _is_v7_model_version(model_version):
        return _v7_execution_signal(
            row=row,
            event_id=event.event_id,
            event_ts=int(event.ts),
            created_at=created_at,
            model_version=model_version,
            snapshot=snapshot,
            family=family,
            round_slug=round_slug,
            token_side=token_side,
            token_by_side=token_by_side,
            round_end_ts=int(round_end_ts),
            bridged_at=bridged_at,
            allowed_outcome_sides=allowed_outcome_sides,
        )

    token_id = str(snapshot.get("source_symbol") or snapshot.get("token_id") or "")
    market = tradable_market_implied_probability(snapshot, event_ts=int(event.ts))
    if not token_id or market is None:
        return None
    prob = float(row["prob_up_15m"])
    token_probability = 1.0 - prob if token_side == "DOWN" else prob
    return ExecutionSignal(
        event_id=event.event_id,
        ts=int(event.ts),
        created_at=created_at,
        model_version=model_version,
        prob_up_15m=prob,
        canonical_symbol=canonical_symbol,
        token_id=token_id,
        outcome_side=token_side,
        round_slug=round_slug,
        round_end_ts=int(round_end_ts),
        market_implied_prob=float(market),
        token_probability=token_probability,
        edge=token_probability - float(market),
        bridged_at=bridged_at,
        opposite_token_id=opposite_token_id,
    )


def _v7_execution_signal(
    *,
    row: dict[str, Any],
    event_id: str,
    event_ts: int,
    created_at: int,
    model_version: str,
    snapshot: dict[str, Any],
    family: str,
    round_slug: str,
    token_side: str,
    token_by_side: dict[tuple[str, str, str], str],
    round_end_ts: int,
    bridged_at: int,
    allowed_outcome_sides: frozenset[str] | None,
) -> ExecutionSignal | None:
    side = _selected_v7_side(snapshot, token_side=token_side)
    if side is None:
        return None
    if allowed_outcome_sides is not None and side not in allowed_outcome_sides:
        return None
    token_id = (
        str(snapshot.get("source_symbol") or snapshot.get("token_id") or "")
        if side == token_side
        else token_by_side.get((family, round_slug, side), "")
    )
    if not token_id:
        return None
    opposite_side = "DOWN" if side == "UP" else "UP"
    opposite_token_id = (
        str(snapshot.get("source_symbol") or snapshot.get("token_id") or "")
        if opposite_side == token_side
        else token_by_side.get((family, round_slug, opposite_side), "")
    )
    p_up = _optional_float(snapshot.get("p_up"))
    p_down = _optional_float(snapshot.get("p_down"))
    if p_up is None or p_down is None:
        return None
    market = _v7_selected_market(snapshot, selected_side=side, token_side=token_side)
    if market is None:
        return None
    polymarket_price = _optional_float(snapshot.get("polymarket_price"))
    if polymarket_price is None:
        polymarket_price = market
    p_up_residual = _optional_float(snapshot.get("p_up_residual_adjusted"))
    p_down_residual = _optional_float(snapshot.get("p_down_residual_adjusted"))
    model_probability = _optional_float(snapshot.get("model_probability"))
    if model_probability is None:
        model_probability = _optional_float(snapshot.get("token_expected_win_probability"))
    if model_probability is None:
        residual_probability = p_up_residual if side == "UP" else p_down_residual
        model_probability = (
            residual_probability
            if residual_probability is not None
            else (p_up if side == "UP" else p_down)
        )
    token_probability = model_probability
    edge = _v7_side_edge(snapshot, side=side, market=market, token_probability=token_probability)
    if edge is None:
        edge = token_probability - market
    entry_worst = _optional_float(
        snapshot.get("entry_worst_price_up")
        if side == "UP"
        else snapshot.get("entry_worst_price_down")
    )
    return ExecutionSignal(
        event_id=event_id,
        ts=event_ts,
        created_at=created_at,
        model_version=model_version,
        prob_up_15m=p_up,
        canonical_symbol=f"{family}:{round_slug}:{side}",
        token_id=str(token_id),
        outcome_side=side,
        round_slug=round_slug,
        round_end_ts=round_end_ts,
        market_implied_prob=market,
        token_probability=token_probability,
        edge=edge,
        bridged_at=bridged_at,
        opposite_token_id=str(opposite_token_id or ""),
        model_probability=model_probability,
        polymarket_price=polymarket_price,
        mispricing_edge=edge,
        p_up=p_up,
        p_down=p_down,
        p_neutral=_optional_float(snapshot.get("p_neutral")),
        p_vol_up=None,
        p_vol_down=None,
        settlement_residual=_optional_float(snapshot.get("settlement_residual")),
        token_expected_win_probability=token_probability,
        p_up_residual_adjusted=p_up_residual,
        p_down_residual_adjusted=p_down_residual,
        expected_edge_up=_optional_float(snapshot.get("expected_edge_up")),
        expected_edge_down=_optional_float(snapshot.get("expected_edge_down")),
        residual_expected_edge_up=_optional_float(snapshot.get("residual_expected_edge_up")),
        residual_expected_edge_down=_optional_float(snapshot.get("residual_expected_edge_down")),
        p_up_hit_5c_before_loss_10c=_optional_float(
            snapshot.get("p_up_hit_5c_before_loss_10c")
        ),
        p_up_hit_10c_before_loss_10c=_optional_float(
            snapshot.get("p_up_hit_10c_before_loss_10c")
        ),
        p_up_loss_10c_before_hit_5c=_optional_float(
            snapshot.get("p_up_loss_10c_before_hit_5c")
        ),
        p_down_hit_5c_before_loss_10c=_optional_float(
            snapshot.get("p_down_hit_5c_before_loss_10c")
        ),
        p_down_hit_10c_before_loss_10c=_optional_float(
            snapshot.get("p_down_hit_10c_before_loss_10c")
        ),
        p_down_loss_10c_before_hit_5c=_optional_float(
            snapshot.get("p_down_loss_10c_before_hit_5c")
        ),
        selected_hit_5c_before_loss_10c=_optional_float(
            snapshot.get("selected_hit_5c_before_loss_10c")
        ),
        selected_hit_10c_before_loss_10c=_optional_float(
            snapshot.get("selected_hit_10c_before_loss_10c")
        ),
        selected_loss_10c_before_hit_5c=_optional_float(
            snapshot.get("selected_loss_10c_before_hit_5c")
        ),
        selected_confidence_score=_optional_float(snapshot.get("selected_confidence_score")),
        selected_side=side,
        selected_expected_edge=edge,
        entry_worst_price=entry_worst,
        should_enter_settlement=_optional_bool(snapshot.get("should_enter_settlement")),
    )


def _is_v7_model_version(model_version: str) -> bool:
    return model_version == "xgboost-v7" or model_version.startswith("xgboost-v7:")


def _selected_v7_side(snapshot: dict[str, Any], *, token_side: str | None = None) -> str | None:
    selected = str(snapshot.get("selected_side") or "").upper()
    if selected in {"UP", "DOWN"}:
        return selected
    if token_side in {"UP", "DOWN"}:
        edge = _v7_side_edge_from_snapshot(snapshot, side=token_side, token_side=token_side)
        if edge is not None:
            return token_side
    up_edge = _v7_side_edge_from_snapshot(snapshot, side="UP", token_side=token_side)
    down_edge = _v7_side_edge_from_snapshot(snapshot, side="DOWN", token_side=token_side)
    if up_edge is not None or down_edge is not None:
        if down_edge is None or (up_edge is not None and up_edge >= down_edge):
            return "UP"
        return "DOWN"
    up_edge = _optional_float(snapshot.get("expected_edge_up"))
    down_edge = _optional_float(snapshot.get("expected_edge_down"))
    if up_edge is None and down_edge is None:
        return None
    if down_edge is None or (up_edge is not None and up_edge >= down_edge):
        return "UP"
    return "DOWN"


def _v7_side_edge(
    snapshot: dict[str, Any],
    *,
    side: str,
    market: float,
    token_probability: float,
) -> float | None:
    mispricing = _optional_float(snapshot.get("mispricing_edge"))
    if mispricing is not None:
        return mispricing
    if _optional_float(snapshot.get("model_probability")) is not None:
        return token_probability - market
    edge = _optional_float(
        snapshot.get("expected_edge_up") if side == "UP" else snapshot.get("expected_edge_down")
    )
    if edge is not None and _optional_float(
        snapshot.get("p_up_residual_adjusted")
        if side == "UP"
        else snapshot.get("p_down_residual_adjusted")
    ) is None:
        return edge
    residual = _optional_float(
        snapshot.get("residual_expected_edge_up")
        if side == "UP"
        else snapshot.get("residual_expected_edge_down")
    )
    if residual is not None:
        return residual
    return token_probability - market


def _v7_side_edge_from_snapshot(
    snapshot: dict[str, Any],
    *,
    side: str,
    token_side: str | None,
) -> float | None:
    mispricing = _optional_float(snapshot.get("mispricing_edge"))
    if mispricing is not None and (token_side is None or side == token_side):
        return mispricing
    model_probability = _optional_float(snapshot.get("model_probability"))
    if model_probability is not None and token_side is not None and side == token_side:
        market = _v7_selected_market(snapshot, selected_side=side, token_side=token_side)
        if market is not None:
            return model_probability - market
    residual_probability = _optional_float(
        snapshot.get("p_up_residual_adjusted")
        if side == "UP"
        else snapshot.get("p_down_residual_adjusted")
    )
    if residual_probability is not None and token_side is not None:
        market = _v7_selected_market(snapshot, selected_side=side, token_side=token_side)
        if market is not None:
            return residual_probability - market
    return None


def _v7_selected_market(
    snapshot: dict[str, Any],
    *,
    selected_side: str,
    token_side: str,
) -> float | None:
    value = _optional_float(snapshot.get("market_implied_prob"))
    if value is None:
        return None
    if selected_side == token_side:
        return value
    return max(0.0, min(1.0, 1.0 - value))


def _snapshot_from_prediction_row(row: dict[str, Any]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "source": row.get("source"),
        "source_symbol": row.get("source_symbol"),
        "source_market": row.get("source_market"),
        "canonical_symbol": row.get("canonical_symbol"),
        "symbol": row.get("symbol"),
        "market_implied_prob": row.get("market_implied_prob"),
        "p_up": row.get("p_up"),
        "p_down": row.get("p_down"),
        "p_neutral": row.get("p_neutral"),
        "p_vol_up": row.get("p_vol_up"),
        "p_vol_down": row.get("p_vol_down"),
        "settlement_residual": row.get("settlement_residual"),
        "token_side": row.get("token_side"),
        "model_probability": row.get("model_probability"),
        "polymarket_price": row.get("polymarket_price"),
        "mispricing_edge": row.get("mispricing_edge"),
        "token_expected_win_probability": row.get("token_expected_win_probability"),
        "p_up_residual_adjusted": row.get("p_up_residual_adjusted"),
        "p_down_residual_adjusted": row.get("p_down_residual_adjusted"),
        "entry_worst_price_up": row.get("entry_worst_price_up"),
        "entry_worst_price_down": row.get("entry_worst_price_down"),
        "expected_edge_up": row.get("expected_edge_up"),
        "expected_edge_down": row.get("expected_edge_down"),
        "residual_expected_edge_up": row.get("residual_expected_edge_up"),
        "residual_expected_edge_down": row.get("residual_expected_edge_down"),
        "p_up_hit_5c_before_loss_10c": row.get("p_up_hit_5c_before_loss_10c"),
        "p_up_hit_10c_before_loss_10c": row.get("p_up_hit_10c_before_loss_10c"),
        "p_up_loss_10c_before_hit_5c": row.get("p_up_loss_10c_before_hit_5c"),
        "p_down_hit_5c_before_loss_10c": row.get("p_down_hit_5c_before_loss_10c"),
        "p_down_hit_10c_before_loss_10c": row.get("p_down_hit_10c_before_loss_10c"),
        "p_down_loss_10c_before_hit_5c": row.get("p_down_loss_10c_before_hit_5c"),
        "selected_hit_5c_before_loss_10c": row.get("selected_hit_5c_before_loss_10c"),
        "selected_hit_10c_before_loss_10c": row.get("selected_hit_10c_before_loss_10c"),
        "selected_loss_10c_before_hit_5c": row.get("selected_loss_10c_before_hit_5c"),
        "selected_confidence_score": row.get("selected_confidence_score"),
        "selected_side": row.get("selected_side"),
        "selected_expected_edge": row.get("selected_expected_edge"),
        "should_enter_settlement": row.get("should_enter_settlement"),
    }
    feature_values_json = row.get("feature_values_json")
    if feature_values_json is not None:
        try:
            features = json.loads(str(feature_values_json))
        except json.JSONDecodeError:
            features = {}
        if isinstance(features, dict):
            for key, value in features.items():
                snapshot.setdefault(key, value)
            snapshot["features"] = features
    return snapshot


def _token_ids_by_market_side(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], str]:
    token_by_side: dict[tuple[str, str, str], str] = {}
    for row in rows:
        snapshot = _snapshot_from_prediction_row(row)
        parsed = _parse_canonical_symbol(
            str(snapshot.get("canonical_symbol") or snapshot.get("symbol") or "")
        )
        if parsed is None:
            continue
        family, round_slug, token_side = parsed
        token_id = str(snapshot.get("source_symbol") or snapshot.get("token_id") or "")
        if token_side in {"UP", "DOWN"} and token_id:
            token_by_side[(family, round_slug, token_side)] = token_id
    return token_by_side


def _parse_canonical_symbol(canonical_symbol: str) -> tuple[str, str, str] | None:
    parts = canonical_symbol.split(":")
    if len(parts) < 3:
        return None
    return parts[0].upper(), parts[-2], parts[-1].upper()


def _current_round_slug(signals: list[ExecutionSignal], *, now_ms: int | None = None) -> str | None:
    """Return the nearest active round represented by the executor signal batch."""

    if not signals:
        raise ValueError("signals is empty")
    now_ms = _now_ms() if now_ms is None else int(now_ms)
    active = [signal for signal in signals if signal.round_end_ts >= now_ms]
    if active:
        nearest = min(active, key=lambda signal: (signal.round_end_ts, -signal.ts))
        return nearest.round_slug
    return None


def _existing_queue_round_slug(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            for raw in reversed(handle.readlines()):
                raw = raw.strip()
                if not raw:
                    continue
                payload = json.loads(raw)
                round_slug = payload.get("round_slug")
                return str(round_slug) if round_slug else None
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _existing_signal_identities(path: Path, round_slug: str) -> set[tuple[object, ...]]:
    return _existing_payload_identities(
        path,
        round_slug,
        payload_identity_fn=_payload_signal_identity,
    )


def _existing_payload_identities(
    path: Path,
    round_slug: str,
    *,
    payload_identity_fn: Any,
) -> set[tuple[object, ...]]:
    if not path.exists():
        return set()
    identities: set[tuple[object, ...]] = set()
    try:
        with path.open(encoding="utf-8") as handle:
            for raw in handle:
                raw = raw.strip()
                if not raw:
                    continue
                payload = json.loads(raw)
                if payload.get("round_slug") != round_slug:
                    continue
                identities.add(payload_identity_fn(payload))
    except (OSError, json.JSONDecodeError):
        return set()
    return identities


def signal_jsonl_prefix_signature(lines: list[str], line_number: int) -> str:
    if line_number <= 0:
        return ""
    h = hashlib.sha256()
    for line in lines[:line_number]:
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _round_signal_counts(signals: list[ExecutionSignal]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for signal in signals:
        counts[signal.round_slug] = counts.get(signal.round_slug, 0) + 1
    return dict(sorted(counts.items()))


@dataclass(frozen=True, slots=True)
class _RawQuote:
    canonical_symbol: str
    bid: float
    ask: float
    ts: int
    published_at_ms: int


def _read_signal_payloads(path: Path) -> list[dict[str, Any]]:
    return JsonlSignalSource(path).read_after(SignalCursor()).payloads


def _current_round_slug_from_payloads(
    payloads: list[dict[str, Any]],
    *,
    now_ms: int,
) -> str | None:
    active: list[dict[str, Any]] = []
    for payload in payloads:
        round_slug = str(payload.get("round_slug") or "")
        round_end_ts = _optional_int(payload.get("round_end_ts"))
        if round_end_ts is None:
            round_end_ts = _round_end_ts_from_round_slug(round_slug)
        if round_slug and round_end_ts is not None and round_end_ts >= now_ms:
            active.append({**payload, "round_end_ts": round_end_ts})
    if not active:
        return None
    nearest = min(active, key=lambda payload: (int(payload["round_end_ts"]), -int(payload.get("ts") or 0)))
    return str(nearest.get("round_slug") or "") or None


def _latest_v7_base_payloads_by_symbol(
    payloads: list[dict[str, Any]],
    *,
    current_round_slug: str,
    now_ms: int,
    max_base_signal_age_seconds: float | None,
    allowed_families: frozenset[str],
) -> dict[str, dict[str, Any]]:
    base_by_symbol: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        if not _is_v7_model_version(str(payload.get("model_version") or "")):
            continue
        if str(payload.get("round_slug") or "") != current_round_slug:
            continue
        canonical_symbol = str(payload.get("canonical_symbol") or "")
        parsed = _parse_canonical_symbol(canonical_symbol)
        if parsed is None:
            continue
        family, _round_slug, token_side = parsed
        if family not in allowed_families or token_side not in {"UP", "DOWN"}:
            continue
        side = str(payload.get("outcome_side") or "").upper()
        if side != token_side:
            continue
        if max_base_signal_age_seconds is not None:
            created_at = _optional_int(payload.get("created_at"))
            bridged_at = _optional_int(payload.get("bridged_at"))
            base_ts = created_at or bridged_at or _optional_int(payload.get("ts")) or 0
            if (now_ms - base_ts) / 1000.0 > max_base_signal_age_seconds:
                continue
        previous = base_by_symbol.get(canonical_symbol)
        if previous is None or int(payload.get("ts") or 0) >= int(previous.get("ts") or 0):
            base_by_symbol[canonical_symbol] = payload
    return base_by_symbol


def _latest_top_of_book_by_bucket(
    items: list[RawQueueItem],
    *,
    current_round_slug: str,
    bucket_seconds: float,
    allowed_families: frozenset[str],
    base_by_symbol: Mapping[str, dict[str, Any]],
) -> dict[tuple[str, int], _RawQuote]:
    bucket_ms = int(bucket_seconds * 1000)
    if bucket_ms <= 0:
        raise ValueError("bucket_seconds must be positive")
    latest: dict[tuple[str, int], _RawQuote] = {}
    for item in items:
        quote = _raw_quote_from_item(item)
        if quote is None:
            continue
        parsed = _parse_canonical_symbol(quote.canonical_symbol)
        if parsed is None:
            continue
        family, round_slug, token_side = parsed
        if family not in allowed_families or round_slug != current_round_slug:
            continue
        if token_side not in {"UP", "DOWN"} or quote.canonical_symbol not in base_by_symbol:
            continue
        bucket_ts = (quote.ts // bucket_ms) * bucket_ms
        key = (quote.canonical_symbol, bucket_ts)
        previous = latest.get(key)
        if previous is None or (quote.ts, quote.published_at_ms) >= (
            previous.ts,
            previous.published_at_ms,
        ):
            latest[key] = quote
    return latest


def _raw_quote_from_item(item: RawQueueItem) -> _RawQuote | None:
    if item.table != "raw_top_of_book":
        return None
    row = item.row
    canonical_symbol = str(row.get("canonical_symbol") or "")
    if not canonical_symbol:
        return None
    bid = _optional_float(row.get("bid_price"))
    ask = _optional_float(row.get("ask_price"))
    if bid is None or ask is None or bid < 0 or ask <= 0 or ask < bid:
        return None
    return _RawQuote(
        canonical_symbol=canonical_symbol,
        bid=bid,
        ask=ask,
        ts=_row_ts_ms(row, item.published_at_ms),
        published_at_ms=int(item.published_at_ms),
    )


def _event_driven_v7_signal_from_quote(
    base: dict[str, Any],
    quote: _RawQuote,
    *,
    bucket_ts: int,
    bridged_at: int,
) -> ExecutionSignal | None:
    side = str(base.get("outcome_side") or "").upper()
    if side not in {"UP", "DOWN"}:
        return None
    token_id = str(base.get("token_id") or base.get("source_symbol") or "")
    if not token_id:
        return None
    p_up = _optional_float(base.get("p_up"))
    p_down = _optional_float(base.get("p_down"))
    p_up_residual = _optional_float(base.get("p_up_residual_adjusted"))
    p_down_residual = _optional_float(base.get("p_down_residual_adjusted"))
    model_probability = _optional_float(base.get("model_probability"))
    if model_probability is None:
        model_probability = _optional_float(base.get("token_expected_win_probability"))
    if model_probability is None:
        model_probability = _optional_float(base.get("token_probability"))
    if model_probability is None and p_up is not None and p_down is not None:
        model_probability = p_up if side == "UP" else p_down
    if model_probability is None:
        return None
    market = quote.ask
    edge = model_probability - market
    expected_edge_up = edge if side == "UP" else None
    expected_edge_down = edge if side == "DOWN" else None
    return ExecutionSignal(
        event_id=(
            f"{base.get('event_id') or 'v7'}-event-driven-"
            f"{bucket_ts}-{quote.published_at_ms}"
        ),
        ts=int(bucket_ts),
        created_at=int(quote.published_at_ms),
        model_version=str(base.get("model_version") or "xgboost-v7"),
        prob_up_15m=0.0 if p_up is None else p_up,
        canonical_symbol=quote.canonical_symbol,
        token_id=token_id,
        outcome_side=side,
        round_slug=str(base.get("round_slug") or ""),
        round_end_ts=int(base.get("round_end_ts") or 0),
        market_implied_prob=market,
        token_probability=model_probability,
        edge=edge,
        bridged_at=int(bridged_at),
        opposite_token_id=str(base.get("opposite_token_id") or ""),
        model_probability=model_probability,
        polymarket_price=market,
        mispricing_edge=edge,
        p_up=p_up,
        p_down=p_down,
        p_neutral=_optional_float(base.get("p_neutral")),
        p_vol_up=_optional_float(base.get("p_vol_up")),
        p_vol_down=_optional_float(base.get("p_vol_down")),
        settlement_residual=_optional_float(base.get("settlement_residual")),
        token_expected_win_probability=model_probability,
        p_up_residual_adjusted=p_up_residual,
        p_down_residual_adjusted=p_down_residual,
        expected_edge_up=expected_edge_up,
        expected_edge_down=expected_edge_down,
        residual_expected_edge_up=expected_edge_up,
        residual_expected_edge_down=expected_edge_down,
        p_up_hit_5c_before_loss_10c=_optional_float(
            base.get("p_up_hit_5c_before_loss_10c")
        ),
        p_up_hit_10c_before_loss_10c=_optional_float(
            base.get("p_up_hit_10c_before_loss_10c")
        ),
        p_up_loss_10c_before_hit_5c=_optional_float(
            base.get("p_up_loss_10c_before_hit_5c")
        ),
        p_down_hit_5c_before_loss_10c=_optional_float(
            base.get("p_down_hit_5c_before_loss_10c")
        ),
        p_down_hit_10c_before_loss_10c=_optional_float(
            base.get("p_down_hit_10c_before_loss_10c")
        ),
        p_down_loss_10c_before_hit_5c=_optional_float(
            base.get("p_down_loss_10c_before_hit_5c")
        ),
        selected_hit_5c_before_loss_10c=_optional_float(
            base.get("selected_hit_5c_before_loss_10c")
        ),
        selected_hit_10c_before_loss_10c=_optional_float(
            base.get("selected_hit_10c_before_loss_10c")
        ),
        selected_loss_10c_before_hit_5c=_optional_float(
            base.get("selected_loss_10c_before_hit_5c")
        ),
        selected_confidence_score=_optional_float(base.get("selected_confidence_score")),
        selected_side=side,
        selected_expected_edge=edge,
        entry_worst_price=market,
        should_enter_settlement=bool(edge > 0),
    )


def _append_event_driven_signals(
    path: Path,
    signals: list[ExecutionSignal],
    *,
    current_round_slug: str,
) -> int:
    return JsonlSignalSink(path).append_round_signals(
        signals,
        current_round_slug=current_round_slug,
        identity_fn=_event_driven_bucket_key,
        payload_identity_fn=_payload_event_driven_bucket_key,
    )


def _event_driven_bucket_key(signal: ExecutionSignal) -> tuple[object, ...]:
    return (
        signal.ts,
        signal.round_slug,
    )


def _payload_event_driven_bucket_key(payload: dict[str, Any]) -> tuple[object, ...]:
    return (
        payload.get("ts"),
        payload.get("round_slug"),
    )


def _existing_event_driven_bucket_keys(
    path: Path,
    round_slug: str,
) -> set[tuple[object, ...]]:
    if not path.exists():
        return set()
    keys: set[tuple[object, ...]] = set()
    try:
        with path.open(encoding="utf-8") as handle:
            for raw in handle:
                raw = raw.strip()
                if not raw:
                    continue
                payload = json.loads(raw)
                if payload.get("round_slug") != round_slug:
                    continue
                keys.add(
                    (
                        payload.get("ts"),
                        payload.get("round_slug"),
                    )
                )
    except (OSError, json.JSONDecodeError):
        return set()
    return keys


def _read_raw_cursor(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
        return int(text) if text else 0
    except (OSError, ValueError):
        return None


def _write_raw_cursor(path: Path, offset: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(int(offset)), encoding="utf-8")


def _row_ts_ms(row: Mapping[str, Any], fallback: int) -> int:
    for key in ("ts", "message_ts", "capture_timestamp_ms", "ingest_ts"):
        value = row.get(key)
        if value is None:
            continue
        try:
            return int(float(value))
        except (TypeError, ValueError):
            continue
    return int(fallback)


def _round_end_ts_from_round_slug(round_slug: str) -> int | None:
    if not round_slug:
        return None
    return round_end_ts_from_canonical_symbol(f"BTC-15M:{round_slug}:UP")


def _signal_identity(signal: ExecutionSignal) -> tuple[object, ...]:
    return (
        signal.ts,
        signal.created_at,
        signal.round_slug,
        signal.outcome_side,
        signal.token_id,
        signal.opposite_token_id,
        _rounded_probability(signal.p_up),
        _rounded_probability(signal.p_down),
        _rounded_probability(signal.p_neutral),
        _rounded_probability(signal.p_vol_up),
        _rounded_probability(signal.p_vol_down),
        _rounded_probability(signal.model_probability),
        _rounded_probability(signal.polymarket_price),
        _rounded_probability(signal.mispricing_edge),
        _rounded_probability(signal.token_expected_win_probability),
        _rounded_probability(signal.p_up_residual_adjusted),
        _rounded_probability(signal.p_down_residual_adjusted),
        _rounded_probability(signal.expected_edge_up),
        _rounded_probability(signal.expected_edge_down),
        _rounded_probability(signal.residual_expected_edge_up),
        _rounded_probability(signal.residual_expected_edge_down),
        _rounded_probability(signal.p_up_hit_5c_before_loss_10c),
        _rounded_probability(signal.p_up_hit_10c_before_loss_10c),
        _rounded_probability(signal.p_up_loss_10c_before_hit_5c),
        _rounded_probability(signal.p_down_hit_5c_before_loss_10c),
        _rounded_probability(signal.p_down_hit_10c_before_loss_10c),
        _rounded_probability(signal.p_down_loss_10c_before_hit_5c),
        _rounded_probability(signal.selected_hit_5c_before_loss_10c),
        _rounded_probability(signal.selected_hit_10c_before_loss_10c),
        _rounded_probability(signal.selected_loss_10c_before_hit_5c),
        _rounded_probability(signal.selected_confidence_score),
        _rounded_probability(signal.selected_expected_edge),
    )


def _kafka_signal_key(signal: ExecutionSignal) -> bytes:
    key = "|".join(
        (
            signal.round_slug,
            signal.outcome_side,
            signal.token_id,
            str(signal.ts),
            str(signal.created_at),
        )
    )
    return key.encode("utf-8")


def _payload_signal_identity(payload: dict[str, Any]) -> tuple[object, ...]:
    return (
        payload.get("ts"),
        payload.get("created_at"),
        payload.get("round_slug"),
        payload.get("outcome_side"),
        payload.get("token_id"),
        payload.get("opposite_token_id") or "",
        _rounded_probability(payload.get("p_up")),
        _rounded_probability(payload.get("p_down")),
        _rounded_probability(payload.get("p_neutral")),
        _rounded_probability(payload.get("p_vol_up")),
        _rounded_probability(payload.get("p_vol_down")),
        _rounded_probability(payload.get("model_probability")),
        _rounded_probability(payload.get("polymarket_price")),
        _rounded_probability(payload.get("mispricing_edge")),
        _rounded_probability(payload.get("token_expected_win_probability")),
        _rounded_probability(payload.get("p_up_residual_adjusted")),
        _rounded_probability(payload.get("p_down_residual_adjusted")),
        _rounded_probability(payload.get("expected_edge_up")),
        _rounded_probability(payload.get("expected_edge_down")),
        _rounded_probability(payload.get("residual_expected_edge_up")),
        _rounded_probability(payload.get("residual_expected_edge_down")),
        _rounded_probability(payload.get("p_up_hit_5c_before_loss_10c")),
        _rounded_probability(payload.get("p_up_hit_10c_before_loss_10c")),
        _rounded_probability(payload.get("p_up_loss_10c_before_hit_5c")),
        _rounded_probability(payload.get("p_down_hit_5c_before_loss_10c")),
        _rounded_probability(payload.get("p_down_hit_10c_before_loss_10c")),
        _rounded_probability(payload.get("p_down_loss_10c_before_hit_5c")),
        _rounded_probability(payload.get("selected_hit_5c_before_loss_10c")),
        _rounded_probability(payload.get("selected_hit_10c_before_loss_10c")),
        _rounded_probability(payload.get("selected_loss_10c_before_hit_5c")),
        _rounded_probability(payload.get("selected_confidence_score")),
        _rounded_probability(payload.get("selected_expected_edge")),
    )


def _signal_age_seconds(signal: ExecutionSignal) -> float:
    return max(0.0, (signal.bridged_at - signal.ts) / 1000.0)


def _load_confluent_kafka() -> Any:
    try:
        import confluent_kafka
    except ImportError as exc:
        raise RuntimeError(
            "Kafka signal transport requires the optional confluent-kafka dependency. "
            "Install with `pip install 'bigan[kafka]'` or install confluent-kafka directly."
        ) from exc
    return confluent_kafka


def _rounded_probability(value: object) -> float | None:
    if value is None:
        return None
    return round(float(value), 12)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
        return None
    return bool(value)


def _now_ms() -> int:
    return int(time.time() * 1000)
