"""Settlement reconciliation for live execution rounds."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import duckdb

from bigan.mlops.registry import DEFAULT_MLOPS_DB_PATH, connect_mlops_db
from bigan.monitoring.drift import (
    ChampionDriftThresholds,
    evaluate_label_hit_rate_drift,
    record_champion_drift_incidents,
)
from bigan.monitoring.events import (
    PredictionOutcome,
    compute_brier_component,
    initialize_monitoring_tables,
    record_prediction_outcome,
)

from .position_manager import Position, PositionManager
from .risk import RiskManager

SettlementResult = Literal["UP", "DOWN", "TIMEOUT"]
SettlementReason = Literal["settled", "early_sell", "timeout", "no_entry"]

EXECUTION_SETTLEMENTS_DDL = """
CREATE TABLE IF NOT EXISTS execution_settlements (
    settlement_id VARCHAR PRIMARY KEY,
    event_id VARCHAR NOT NULL,
    model_version VARCHAR NOT NULL,
    side VARCHAR,
    entry_price DOUBLE,
    fill_price DOUBLE,
    exit_price DOUBLE,
    settlement_result VARCHAR NOT NULL CHECK (settlement_result IN ('UP', 'DOWN', 'TIMEOUT')),
    realized_pnl DOUBLE NOT NULL,
    settlement_ts BIGINT NOT NULL,
    reason VARCHAR NOT NULL CHECK (reason IN ('settled', 'early_sell', 'timeout', 'no_entry')),
    details_json VARCHAR NOT NULL,
    created_at BIGINT NOT NULL
)
"""


class SettlementResultProvider(Protocol):
    """Minimal API required to fetch a Polymarket round result."""

    def get_settlement_result(self, event_id: str) -> str | None:
        """Return UP, DOWN, or None while the market is not resolved yet."""


@dataclass(frozen=True, slots=True)
class ExecutionSettlementRecord:
    """One reconciled execution settlement row."""

    settlement_id: str
    event_id: str
    model_version: str
    side: str | None
    entry_price: float | None
    fill_price: float | None
    exit_price: float | None
    settlement_result: SettlementResult
    realized_pnl: float
    settlement_ts: int
    reason: SettlementReason
    details_json: str
    created_at: int | None = None

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["created_at"] = self.created_at or _now_ms()
        return row


@dataclass(frozen=True, slots=True)
class SettlementReconciliationResult:
    """Result from reconciling one event."""

    record: ExecutionSettlementRecord
    position: Position | None
    monitoring_outcome_written: bool
    label_hit_rate: dict[str, Any] | None
    incident_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SettlementPollerConfig:
    """Polling behavior for delayed Polymarket settlements."""

    poll_interval_seconds: float = 30.0
    max_attempts: int = 10


class SettlementPoller:
    """Reconcile position, settlement, risk, and monitoring state."""

    def __init__(
        self,
        *,
        position_manager: PositionManager,
        result_provider: SettlementResultProvider,
        model_version: str,
        db_path: Path | str = DEFAULT_MLOPS_DB_PATH,
        conn: duckdb.DuckDBPyConnection | None = None,
        risk_manager: RiskManager | None = None,
        config: SettlementPollerConfig | None = None,
        label_thresholds: ChampionDriftThresholds | None = None,
        sleep: Any = time.sleep,
    ) -> None:
        self.position_manager = position_manager
        self.result_provider = result_provider
        self.model_version = model_version
        self.db_path = db_path
        self.conn = conn
        self.risk_manager = risk_manager
        self.config = config or SettlementPollerConfig()
        self.label_thresholds = label_thresholds
        self.sleep = sleep
        _validate_config(self.config)
        with self._connection() as active:
            initialize_settlement_tables(active)

    def reconcile_event(
        self,
        event_id: str,
        *,
        settlement_ts: int | None = None,
        update_monitoring: bool = True,
    ) -> SettlementReconciliationResult:
        """Poll and reconcile one round/event settlement."""

        _require_text("event_id", event_id)
        ts = settlement_ts or _now_ms()
        position = self.position_manager.get_position(event_id)
        if position is not None and position.status == "closed":
            record = _early_sell_record(
                event_id=event_id,
                model_version=self.model_version,
                position=position,
                settlement_ts=ts,
            )
            return self._finalize(record, position, update_monitoring=False)

        result = self._poll_result(event_id)
        if result is None:
            record = _timeout_record(
                event_id=event_id,
                model_version=self.model_version,
                position=position,
                settlement_ts=ts,
            )
            return self._finalize(record, position, update_monitoring=False)

        if position is None:
            record = _no_entry_record(
                event_id=event_id,
                model_version=self.model_version,
                settlement_result=result,
                settlement_ts=ts,
            )
            return self._finalize(record, None, update_monitoring=update_monitoring)

        settled_position = self.position_manager.settle_position(
            event_id,
            result,
            settlement_time=ts,
        )
        record = _settled_record(
            event_id=event_id,
            model_version=self.model_version,
            position=position,
            settlement_result=result,
            settlement_ts=ts,
        )
        return self._finalize(record, settled_position, update_monitoring=update_monitoring)

    def _poll_result(self, event_id: str) -> SettlementResult | None:
        for attempt in range(self.config.max_attempts):
            result = _normalise_result(self.result_provider.get_settlement_result(event_id))
            if result is not None:
                return result
            if attempt < self.config.max_attempts - 1:
                self.sleep(self.config.poll_interval_seconds)
        return None

    def _finalize(
        self,
        record: ExecutionSettlementRecord,
        position: Position | None,
        *,
        update_monitoring: bool,
    ) -> SettlementReconciliationResult:
        with self._connection() as conn:
            record_execution_settlement(conn, record, replace=True)
            monitoring_written = False
            label_report = None
            incident_ids: tuple[str, ...] = ()
            if update_monitoring and record.settlement_result in {"UP", "DOWN"}:
                monitoring_written = _record_prediction_outcome_from_settlement(
                    conn,
                    record,
                    position=position,
                    replace=True,
                )
                label_report = evaluate_label_hit_rate_drift(
                    conn,
                    model_version=self.model_version,
                    thresholds=self.label_thresholds,
                )
                if label_report["alert"]:
                    incident_ids = record_champion_drift_incidents(
                        conn,
                        _label_alert_report(
                            model_version=self.model_version,
                            label_report=label_report,
                            generated_at_ms=record.settlement_ts,
                        ),
                    )
        if self.risk_manager is not None and record.reason in {"settled", "early_sell"}:
            self.risk_manager.record_loss(record.realized_pnl)
        return SettlementReconciliationResult(
            record=record,
            position=position,
            monitoring_outcome_written=monitoring_written,
            label_hit_rate=label_report,
            incident_ids=incident_ids,
        )

    def _connection(self) -> Any:
        if self.conn is not None:
            return _BorrowedConnection(self.conn)
        return connect_mlops_db(self.db_path)


def initialize_settlement_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Create settlement reconciliation tables."""

    conn.execute(EXECUTION_SETTLEMENTS_DDL)


def record_execution_settlement(
    conn: duckdb.DuckDBPyConnection,
    record: ExecutionSettlementRecord,
    *,
    replace: bool = False,
) -> None:
    """Insert one execution settlement row."""

    initialize_settlement_tables(conn)
    _validate_record(record)
    row = record.to_row()
    columns = tuple(row)
    placeholders = ", ".join("?" for _ in columns)
    conn.execute(
        f"INSERT {'OR REPLACE ' if replace else ''}INTO execution_settlements "
        f"({', '.join(columns)}) VALUES ({placeholders})",
        [row[column] for column in columns],
    )


def read_execution_settlements(
    conn: duckdb.DuckDBPyConnection,
    *,
    event_id: str | None = None,
) -> list[dict[str, Any]]:
    """Read settlement reconciliation rows."""

    initialize_settlement_tables(conn)
    params: list[str] = []
    predicate = ""
    if event_id is not None:
        predicate = "WHERE event_id = ?"
        params.append(event_id)
    rows = conn.execute(
        f"""
        SELECT *
        FROM execution_settlements
        {predicate}
        ORDER BY settlement_ts ASC, event_id ASC
        """,
        params,
    ).fetchall()
    columns = [column[0] for column in conn.description]
    return [dict(zip(columns, row, strict=True)) for row in rows]


def realized_pnl_for_position(position: Position, settlement_result: str | None = None) -> float:
    """Return realized PnL from fill/exit/settlement state."""

    cost_basis = _cost_basis(position)
    if position.exit_price is not None:
        return (position.exit_price - cost_basis) * position.size
    result = _normalise_result(settlement_result)
    if result is None:
        raise ValueError("settlement_result must be UP or DOWN for open positions")
    payout = 1.0 if result == position.side else 0.0
    return (payout - cost_basis) * position.size


def _record_prediction_outcome_from_settlement(
    conn: duckdb.DuckDBPyConnection,
    record: ExecutionSettlementRecord,
    *,
    position: Position | None,
    replace: bool,
) -> bool:
    initialize_monitoring_tables(conn)
    event = conn.execute(
        """
        SELECT ts, prob_up_15m
        FROM prediction_events
        WHERE event_id = ?
          AND model_version = ?
        """,
        [record.event_id, record.model_version],
    ).fetchone()
    if event is None:
        return False
    realized_label = _settlement_positive_label(record, position=position)
    probability = _prediction_probability(float(event[1]), position=position)
    record_prediction_outcome(
        conn,
        PredictionOutcome(
            event_id=record.event_id,
            target_ts=record.settlement_ts,
            realized_label=realized_label,
            realized_return=record.realized_pnl,
            brier_component=compute_brier_component(probability, realized_label),
            outcome_ts=record.settlement_ts,
        ),
        replace=replace,
    )
    return True


def _settlement_positive_label(
    record: ExecutionSettlementRecord,
    *,
    position: Position | None,
) -> bool:
    if position is None or position.side is None:
        return record.settlement_result == "UP"
    return record.settlement_result == position.side


def _prediction_probability(prob_up_15m: float, *, position: Position | None) -> float:
    if position is not None and position.side == "DOWN":
        return 1.0 - prob_up_15m
    return prob_up_15m


def _label_alert_report(
    *,
    model_version: str,
    label_report: dict[str, Any],
    generated_at_ms: int,
) -> dict[str, Any]:
    threshold = label_report["threshold"]
    return {
        "model_version": model_version,
        "generated_at_ms": generated_at_ms,
        "alerts": [
            {
                "alert_type": "label_hit_rate_low",
                "window_ms": None,
                "severity": "critical",
                "detail": (
                    f"positive_rate={label_report['positive_rate']:.6f} below "
                    f"{threshold:.6f} for {label_report['sample_count']} settled samples"
                ),
            }
        ],
    }


def _settled_record(
    *,
    event_id: str,
    model_version: str,
    position: Position,
    settlement_result: SettlementResult,
    settlement_ts: int,
) -> ExecutionSettlementRecord:
    exit_price = 1.0 if settlement_result == position.side else 0.0
    return _record(
        event_id=event_id,
        model_version=model_version,
        position=position,
        exit_price=exit_price,
        settlement_result=settlement_result,
        realized_pnl=realized_pnl_for_position(position, settlement_result),
        settlement_ts=settlement_ts,
        reason="settled",
    )


def _early_sell_record(
    *,
    event_id: str,
    model_version: str,
    position: Position,
    settlement_ts: int,
) -> ExecutionSettlementRecord:
    return _record(
        event_id=event_id,
        model_version=model_version,
        position=position,
        exit_price=position.exit_price,
        settlement_result="UP" if position.side == "UP" else "DOWN",
        realized_pnl=realized_pnl_for_position(position),
        settlement_ts=settlement_ts,
        reason="early_sell",
    )


def _timeout_record(
    *,
    event_id: str,
    model_version: str,
    position: Position | None,
    settlement_ts: int,
) -> ExecutionSettlementRecord:
    return _record(
        event_id=event_id,
        model_version=model_version,
        position=position,
        exit_price=position.exit_price if position is not None else None,
        settlement_result="TIMEOUT",
        realized_pnl=0.0,
        settlement_ts=settlement_ts,
        reason="timeout",
    )


def _no_entry_record(
    *,
    event_id: str,
    model_version: str,
    settlement_result: SettlementResult,
    settlement_ts: int,
) -> ExecutionSettlementRecord:
    return _record(
        event_id=event_id,
        model_version=model_version,
        position=None,
        exit_price=None,
        settlement_result=settlement_result,
        realized_pnl=0.0,
        settlement_ts=settlement_ts,
        reason="no_entry",
    )


def _record(
    *,
    event_id: str,
    model_version: str,
    position: Position | None,
    exit_price: float | None,
    settlement_result: SettlementResult,
    realized_pnl: float,
    settlement_ts: int,
    reason: SettlementReason,
) -> ExecutionSettlementRecord:
    side = None if position is None else position.side
    details = {
        "order_id": None if position is None else position.order_id,
        "position_status": None if position is None else position.status,
        "size": None if position is None else position.size,
    }
    return ExecutionSettlementRecord(
        settlement_id=f"{event_id}:{reason}",
        event_id=event_id,
        model_version=model_version,
        side=side,
        entry_price=None if position is None else position.entry_price,
        fill_price=None if position is None else _cost_basis(position),
        exit_price=exit_price,
        settlement_result=settlement_result,
        realized_pnl=float(realized_pnl),
        settlement_ts=settlement_ts,
        reason=reason,
        details_json=json.dumps(details, sort_keys=True),
        created_at=settlement_ts,
    )


class _BorrowedConnection:
    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self.conn = conn

    def __enter__(self) -> duckdb.DuckDBPyConnection:
        return self.conn

    def __exit__(self, *args: object) -> None:
        return None


def _normalise_result(value: str | None) -> SettlementResult | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if text in {"UP", "DOWN"}:
        return text  # type: ignore[return-value]
    return None


def _cost_basis(position: Position) -> float:
    return position.fill_price if position.fill_price is not None else position.entry_price


def _validate_config(config: SettlementPollerConfig) -> None:
    if config.poll_interval_seconds < 0:
        raise ValueError("poll_interval_seconds must be non-negative")
    if config.max_attempts <= 0:
        raise ValueError("max_attempts must be positive")


def _validate_record(record: ExecutionSettlementRecord) -> None:
    _require_text("settlement_id", record.settlement_id)
    _require_text("event_id", record.event_id)
    _require_text("model_version", record.model_version)
    if record.side is not None and record.side not in {"UP", "DOWN"}:
        raise ValueError("side must be UP, DOWN, or None")
    if record.settlement_result not in {"UP", "DOWN", "TIMEOUT"}:
        raise ValueError("settlement_result must be UP, DOWN, or TIMEOUT")
    if record.reason not in {"settled", "early_sell", "timeout", "no_entry"}:
        raise ValueError("invalid settlement reason")
    if record.settlement_ts < 0:
        raise ValueError("settlement_ts must be non-negative")
    json.loads(record.details_json)


def _require_text(field_name: str, value: str) -> None:
    if not str(value).strip():
        raise ValueError(f"{field_name} is required")


def _now_ms() -> int:
    return int(time.time() * 1000)
