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


#: Placeholder substituted into the quarantine row when the offending row
#: is missing its symbol identity. Keeps the schema's NOT NULL contract on
#: ``source_symbol`` intact while still allowing the rule to fire.
UNKNOWN_SYMBOL = "<unknown>"


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

    def __init__(self) -> None:
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
        errors.extend(_check_identity(row))

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

        source = str(row.get("source") or "unknown")
        source_symbol = str(row.get("source_symbol") or "").strip() or UNKNOWN_SYMBOL
        source_market = row.get("source_market")
        canonical_symbol = row.get("canonical_symbol")

        provenance = row.get("provenance")

        out: list[dict[str, Any]] = []
        for err in errors:
            out.append(
                {
                    "ts": ts,
                    "message_ts": message_ts,
                    "ingest_ts": ingest_ts,
                    "source": source,
                    "source_symbol": source_symbol,
                    "source_market": source_market,
                    "canonical_symbol": canonical_symbol,
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


def _check_identity(row: dict[str, Any]) -> list[ValidationError]:
    errors: list[ValidationError] = []

    symbol = row.get("source_symbol")
    if symbol is None or (isinstance(symbol, str) and not symbol.strip()):
        errors.append(
            ValidationError(
                rule=ValidationRule.EMPTY_SYMBOL,
                detail="source_symbol is null or empty",
            )
        )

    ts = row.get("ts")
    if ts is None or _int_or_zero(ts) <= 0:
        errors.append(
            ValidationError(
                rule=ValidationRule.EMPTY_TIME,
                detail=f"ts is null or non-positive: {ts!r}",
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
