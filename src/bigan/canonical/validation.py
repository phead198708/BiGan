"""Per-row validation + quarantine routing (issue #4).

The ETL pipeline ingests semi-trusted exchange payloads. Before a canonical
row lands in a ``raw_*`` table we run it through this validation layer.
A row that fails **any** rule is re-routed into the ``quarantine`` table
along with a machine-readable ``rule`` tag and a JSON dump of the
original payload, so:

- the failure can be triaged offline without re-running ingest,
- the main raw tables remain free of corrupted / hostile inputs,
- downstream feature jobs can rely on the raw tables' invariants.

Rules implemented:

- ``crossed_book``      — ``bid_price > ask_price`` on top-of-book rows
- ``negative_size``     — orderbook level or trade with size < 0
- ``negative_price``    — orderbook level or trade with price < 0
- ``empty_symbol``      — ``source_symbol`` missing / empty string
- ``empty_time``        — ``ts`` missing / non-positive
- ``duplicate_trade_id``— ``trade_id`` already seen in this validator instance
- ``ts_in_future``      — ``ts > ingest_ts + future_grace`` (issue #23)
- ``ts_too_stale``      — ``ingest_ts - ts > stale_threshold`` (issue #23)

The validator is **stateful** w.r.t. trade-id dedup: callers should reuse a
single :class:`RowValidator` for the lifetime of an ETL batch so duplicates
are caught across files within the run. Cross-batch dedup is out of scope
for #4 (a future migration may persist seen trade-ids to disk).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import orjson

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rule names
# ---------------------------------------------------------------------------


class ValidationRule(StrEnum):
    CROSSED_BOOK = "crossed_book"
    NEGATIVE_SIZE = "negative_size"
    NEGATIVE_PRICE = "negative_price"
    EMPTY_SYMBOL = "empty_symbol"
    EMPTY_TIME = "empty_time"
    DUPLICATE_TRADE_ID = "duplicate_trade_id"
    # Timestamp Contract (issue #23) — see docs/adr/0002-timestamp-contract.md
    TS_IN_FUTURE = "ts_in_future"
    TS_TOO_STALE = "ts_too_stale"


#: Placeholder substituted into the quarantine row when the offending row
#: is missing its symbol identity. Keeps the schema's NOT NULL contract on
#: ``source_symbol`` intact while still allowing the rule to fire.
UNKNOWN_SYMBOL = "<unknown>"


# Default thresholds for the Timestamp Contract checks (issue #23). Both are
# wallclock-millisecond magnitudes; ETL callers may override via
# :class:`RowValidator` constructor arguments.
DEFAULT_TS_FUTURE_GRACE_MS = 5_000        # 5s tolerance for upstream clock skew
DEFAULT_TS_STALE_THRESHOLD_MS = 600_000   # 10min — beyond this, treat as replay


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class ValidationError:
    """One rule violation discovered on a row."""

    rule: ValidationRule
    detail: str


@dataclass(slots=True)
class ValidationStats:
    """Counters aggregated across one validator's lifetime."""

    rows_checked: dict[str, int] = field(default_factory=dict)
    rows_quarantined_by_rule: dict[str, int] = field(default_factory=dict)

    def record_check(self, table: str) -> None:
        self.rows_checked[table] = self.rows_checked.get(table, 0) + 1

    def record_quarantine(self, rule: ValidationRule) -> None:
        key = rule.value
        self.rows_quarantined_by_rule[key] = self.rows_quarantined_by_rule.get(key, 0) + 1

    @property
    def total_quarantined(self) -> int:
        return sum(self.rows_quarantined_by_rule.values())


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class RowValidator:
    """Stateful row validator that returns quarantine rows for bad inputs.

    Usage::

        validator = RowValidator()
        for row in transformed_rows:
            errors = validator.validate("raw_top_of_book", row)
            if errors:
                q_rows = validator.to_quarantine_rows("raw_top_of_book", row, errors)
                writer.append_rows("quarantine", q_rows)
            else:
                writer.append_rows("raw_top_of_book", [row])
    """

    def __init__(
        self,
        *,
        future_grace_ms: int = DEFAULT_TS_FUTURE_GRACE_MS,
        stale_threshold_ms: int = DEFAULT_TS_STALE_THRESHOLD_MS,
    ) -> None:
        """Construct a validator.

        Args:
            future_grace_ms: Tolerance (ms) for ``ts`` to lead ``ingest_ts``
                before the row is quarantined as ``ts_in_future``. Default
                5s, sized for typical NTP-corrected exchange clock skew.
            stale_threshold_ms: Maximum (ms) that ``ingest_ts`` may lag
                ``ts`` before the row is quarantined as ``ts_too_stale``.
                Default 10min, lets normal #5 backfills through but
                catches accidental replay of week-old data.
        """
        if future_grace_ms < 0:
            raise ValueError("future_grace_ms must be non-negative")
        if stale_threshold_ms < 0:
            raise ValueError("stale_threshold_ms must be non-negative")
        self._future_grace_ms = future_grace_ms
        self._stale_threshold_ms = stale_threshold_ms
        self._seen_trade_ids: set[str] = set()
        self.stats = ValidationStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, table: str, row: dict[str, Any]) -> list[ValidationError]:
        """Return the list of rule violations for ``row``.

        Empty list ⇒ row is clean and should be written to its target table.
        Non-empty ⇒ caller should redirect it to ``quarantine``.
        """
        self.stats.record_check(table)
        errors: list[ValidationError] = []

        # Identity rules apply to every raw_* table.
        errors.extend(
            _check_identity(
                row,
                future_grace_ms=self._future_grace_ms,
                stale_threshold_ms=self._stale_threshold_ms,
            )
        )

        if table == "raw_top_of_book":
            errors.extend(_check_top_of_book(row))
        elif table == "raw_orderbook_snapshot":
            errors.extend(_check_orderbook_snapshot(row))
        elif table == "raw_trades":
            errors.extend(_check_trade(row, self._seen_trade_ids))

        for err in errors:
            self.stats.record_quarantine(err.rule)
        return errors

    def to_quarantine_rows(
        self,
        target_table: str,
        row: dict[str, Any],
        errors: list[ValidationError],
    ) -> list[dict[str, Any]]:
        """Convert one bad row + its errors into ``quarantine`` rows.

        We emit **one quarantine row per rule** so downstream analysts can
        ``GROUP BY rule`` without a sub-aggregation. The original row is
        JSON-encoded into ``payload_json`` so nothing is lost.
        """
        if not errors:
            return []

        payload_json = orjson.dumps(_jsonable(row)).decode("utf-8")

        ingest_ts = _int_or_zero(row.get("ingest_ts"))
        ts = _int_or_zero(row.get("ts")) or ingest_ts
        message_ts = _int_or_zero(row.get("message_ts")) or ts
        capture_timestamp_ms = _int_or_zero(row.get("capture_timestamp_ms")) or None

        source = str(row.get("source") or "unknown")
        source_symbol = str(row.get("source_symbol") or "").strip() or UNKNOWN_SYMBOL
        source_market = row.get("source_market")
        canonical_symbol = row.get("canonical_symbol")
        source_channel = row.get("source_channel")

        provenance = row.get("provenance")

        out: list[dict[str, Any]] = []
        for err in errors:
            out.append(
                {
                    "ts": ts,
                    "message_ts": message_ts,
                    "ingest_ts": ingest_ts,
                    "capture_timestamp_ms": capture_timestamp_ms,
                    "source": source,
                    "source_symbol": source_symbol,
                    "source_market": source_market,
                    "canonical_symbol": canonical_symbol,
                    "source_channel": source_channel,
                    "provenance": provenance,
                    "target_table": target_table,
                    "rule": err.rule.value,
                    "detail": err.detail,
                    "payload_json": payload_json,
                }
            )
        return out


# ---------------------------------------------------------------------------
# Per-rule helpers
# ---------------------------------------------------------------------------


def _check_identity(
    row: dict[str, Any],
    *,
    future_grace_ms: int,
    stale_threshold_ms: int,
) -> list[ValidationError]:
    errors: list[ValidationError] = []

    symbol = row.get("source_symbol")
    if symbol is None or (isinstance(symbol, str) and not symbol.strip()):
        errors.append(
            ValidationError(
                rule=ValidationRule.EMPTY_SYMBOL,
                detail="source_symbol is null or empty",
            )
        )

    ts_raw = row.get("ts")
    ts_val = _int_or_zero(ts_raw)
    if ts_raw is None or ts_val <= 0:
        errors.append(
            ValidationError(
                rule=ValidationRule.EMPTY_TIME,
                detail=f"ts is null or non-positive: {ts_raw!r}",
            )
        )
        # Bail out of the temporal sanity checks if we have no usable ts.
        return errors

    # Timestamp Contract checks (issue #23). We require a positive
    # ``ingest_ts`` to evaluate either direction; if it's missing we
    # leave it to upstream observability rather than over-quarantining.
    ingest_ts = _int_or_zero(row.get("ingest_ts"))
    if ingest_ts <= 0:
        return errors

    if ts_val - ingest_ts > future_grace_ms:
        errors.append(
            ValidationError(
                rule=ValidationRule.TS_IN_FUTURE,
                detail=(
                    f"ts ({ts_val}) exceeds ingest_ts ({ingest_ts}) by "
                    f"{ts_val - ingest_ts}ms (grace={future_grace_ms}ms)"
                ),
            )
        )
    elif ingest_ts - ts_val > stale_threshold_ms:
        # Only raise stale if not already in_future (same row can't be both).
        errors.append(
            ValidationError(
                rule=ValidationRule.TS_TOO_STALE,
                detail=(
                    f"ingest_ts ({ingest_ts}) lags ts ({ts_val}) by "
                    f"{ingest_ts - ts_val}ms (threshold={stale_threshold_ms}ms)"
                ),
            )
        )
    return errors


def _check_top_of_book(row: dict[str, Any]) -> list[ValidationError]:
    bid = row.get("bid_price")
    ask = row.get("ask_price")
    errors: list[ValidationError] = []
    if bid is not None and bid < 0:
        errors.append(
            ValidationError(
                rule=ValidationRule.NEGATIVE_PRICE,
                detail=f"bid_price < 0: {bid!r}",
            )
        )
    if ask is not None and ask < 0:
        errors.append(
            ValidationError(
                rule=ValidationRule.NEGATIVE_PRICE,
                detail=f"ask_price < 0: {ask!r}",
            )
        )
    if bid is not None and ask is not None and bid > ask:
        errors.append(
            ValidationError(
                rule=ValidationRule.CROSSED_BOOK,
                detail=f"bid_price ({bid}) > ask_price ({ask})",
            )
        )
    return errors


def _check_orderbook_snapshot(row: dict[str, Any]) -> list[ValidationError]:
    errors: list[ValidationError] = []
    size = row.get("size")
    if size is not None and size < 0:
        errors.append(
            ValidationError(
                rule=ValidationRule.NEGATIVE_SIZE,
                detail=f"size < 0: {size!r}",
            )
        )
    price = row.get("price")
    if price is not None and price < 0:
        errors.append(
            ValidationError(
                rule=ValidationRule.NEGATIVE_PRICE,
                detail=f"price < 0: {price!r}",
            )
        )
    return errors


def _check_trade(row: dict[str, Any], seen: set[str]) -> list[ValidationError]:
    errors: list[ValidationError] = []

    size = row.get("size")
    if size is not None and size < 0:
        errors.append(
            ValidationError(
                rule=ValidationRule.NEGATIVE_SIZE,
                detail=f"size < 0: {size!r}",
            )
        )
    price = row.get("price")
    if price is not None and price < 0:
        errors.append(
            ValidationError(
                rule=ValidationRule.NEGATIVE_PRICE,
                detail=f"price < 0: {price!r}",
            )
        )

    trade_id = row.get("trade_id")
    if trade_id:
        tid = str(trade_id)
        if tid in seen:
            errors.append(
                ValidationError(
                    rule=ValidationRule.DUPLICATE_TRADE_ID,
                    detail=f"trade_id already seen: {tid!r}",
                )
            )
        else:
            seen.add(tid)
    return errors


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def _int_or_zero(value: Any) -> int:
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0


def _jsonable(value: Any) -> Any:
    """Make a row dict orjson-encodable by stringifying unknown types."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)
