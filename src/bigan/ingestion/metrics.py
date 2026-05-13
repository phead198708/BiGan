"""Prometheus metrics registry for the ingestion service.

Exposes counters / gauges / histograms that operators can scrape:

- ``bigan_ws_messages_total{event_type=..}`` — total messages received.
- ``bigan_ws_parse_errors_total{kind=..}`` — parse / validation failures.
- ``bigan_ws_reconnects_total`` — full reconnect cycles.
- ``bigan_ws_subscribed_markets`` — current subscription set size (gauge).
- ``bigan_ws_hash_mismatch_total`` — book/delta hash inconsistencies.
- ``bigan_sink_records_written_total`` — successful raw writes.
- ``bigan_sink_flush_seconds`` — flush latency histogram.
- ``bigan_last_event_receive_time_seconds`` — gauge of the latest receive time
  (used for liveness alarms).
- ``bigan_ingest_lag_seconds{source=..,event_type=..}`` — local receive time
  minus upstream message timestamp.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry()

WS_MESSAGES_TOTAL = Counter(
    "bigan_ws_messages_total",
    "Total CLOB WebSocket messages received, partitioned by event_type.",
    labelnames=("event_type",),
    registry=REGISTRY,
)

WS_PARSE_ERRORS_TOTAL = Counter(
    "bigan_ws_parse_errors_total",
    "Messages that could not be decoded or validated.",
    labelnames=("kind",),
    registry=REGISTRY,
)

WS_RECONNECTS_TOTAL = Counter(
    "bigan_ws_reconnects_total",
    "Number of full WebSocket reconnect cycles.",
    registry=REGISTRY,
)

WS_SUBSCRIBED_MARKETS = Gauge(
    "bigan_ws_subscribed_markets",
    "Number of asset_ids currently subscribed on the active connection.",
    registry=REGISTRY,
)

WS_HASH_MISMATCH_TOTAL = Counter(
    "bigan_ws_hash_mismatch_total",
    "Times a price_change delta failed hash verification against the local book.",
    labelnames=("asset_id",),
    registry=REGISTRY,
)

SINK_RECORDS_WRITTEN_TOTAL = Counter(
    "bigan_sink_records_written_total",
    "Records successfully written to the raw sink.",
    registry=REGISTRY,
)

SINK_FLUSH_SECONDS = Histogram(
    "bigan_sink_flush_seconds",
    "Latency of sink flush() operations.",
    registry=REGISTRY,
)

LAST_EVENT_RECEIVE_TIME = Gauge(
    "bigan_last_event_receive_time_seconds",
    "Receive time (epoch s) of the most recent message; used for liveness alarms.",
    registry=REGISTRY,
)

INGEST_LAG_SECONDS = Histogram(
    "bigan_ingest_lag_seconds",
    "Delta between local receive time and upstream message timestamp.",
    labelnames=("source", "event_type"),
    buckets=(0.001, 0.005, 0.010, 0.025, 0.050, 0.100, 0.250, 0.500, 1.0, 2.5, 5.0, 10.0),
    registry=REGISTRY,
)

GAMMA_POLLS_TOTAL = Counter(
    "bigan_gamma_polls_total",
    "Total Gamma API poll attempts.",
    labelnames=("outcome",),
    registry=REGISTRY,
)

ROLLUP_FILES_TOTAL = Counter(
    "bigan_rollup_files_total",
    "Files converted from NDJSON.gz to Parquet by the rollup worker.",
    labelnames=("outcome",),
    registry=REGISTRY,
)

# --- Gap detection / REST backfill (issue #5) ---

GAP_DETECTED_TOTAL = Counter(
    "bigan_gap_detected_total",
    "Per-asset stream-silence detections (gap entered).",
    labelnames=("asset_id",),
    registry=REGISTRY,
)

GAP_RESOLVED_TOTAL = Counter(
    "bigan_gap_resolved_total",
    "Per-asset stream-silence resolutions (gap exited via resume).",
    labelnames=("asset_id",),
    registry=REGISTRY,
)

GAP_SILENCE_DURATION_SECONDS = Histogram(
    "bigan_gap_silence_duration_seconds",
    "Distribution of resolved gap silence durations.",
    registry=REGISTRY,
)

BACKFILL_INVOCATIONS_TOTAL = Counter(
    "bigan_backfill_invocations_total",
    "Backfill service invocations partitioned by outcome.",
    labelnames=("outcome",),
    registry=REGISTRY,
)

BACKFILL_RECORDS_TOTAL = Counter(
    "bigan_backfill_records_total",
    "Records replayed by the backfill service (synthetic NDJSON writes).",
    labelnames=("kind",),
    registry=REGISTRY,
)
