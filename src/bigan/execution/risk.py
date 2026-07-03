"""Capital-control gates for live champion execution."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import duckdb

from bigan.monitoring.incidents import DataQualityIncident, record_data_quality_incident

from .position_manager import PositionManager


@dataclass(frozen=True, slots=True)
class RiskConfig:
    """Risk and sizing configuration for live execution."""

    max_position_size_usdc: float = 10.0
    max_concurrent_positions: int = 2
    daily_loss_limit_usdc: float = 50.0
    max_drawdown_pct: float = 0.20
    kelly_fraction: float = 0.10
    min_edge_for_full_size: float = 0.50
    available_budget_usdc: float = 100.0
    min_order_size_usdc: float = 1.0
    starting_equity_usdc: float = 100.0


@dataclass(frozen=True, slots=True)
class DailyRiskStats:
    """One UTC-day risk accounting snapshot."""

    date: str
    realized_pnl_usdc: float
    trade_count: int
    remaining_loss_budget_usdc: float
    current_equity_usdc: float
    peak_equity_usdc: float
    drawdown_pct: float
    circuit_breaker_active: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EntryRiskDecision:
    """Risk decision for one potential entry."""

    allowed: bool
    size: float
    reason: str

    def __iter__(self) -> Any:
        yield self.allowed
        yield self.size
        yield self.reason


class RiskManager:
    """Apply daily loss, drawdown, concurrency, and size gates before entry."""

    def __init__(
        self,
        position_manager: PositionManager,
        *,
        config: RiskConfig | None = None,
        incident_conn: duckdb.DuckDBPyConnection | None = None,
    ) -> None:
        self.position_manager = position_manager
        self.config = config or RiskConfig()
        self._incident_conn = incident_conn
        self._date = _utc_date()
        self._realized_pnl = 0.0
        self._trade_count = 0
        self._peak_equity = self.config.starting_equity_usdc
        self._circuit_breaker_reason: str | None = None
        _validate_config(self.config)

    def check_entry_allowed(self, event_id: str, edge: float) -> EntryRiskDecision:
        """Return whether a BUY entry is allowed and the recommended USDC size."""

        self._reset_if_new_day()
        if self._circuit_breaker_reason is not None:
            return EntryRiskDecision(False, 0.0, self._circuit_breaker_reason)
        if self.position_manager.has_open_position(event_id):
            return EntryRiskDecision(False, 0.0, "position_already_open")
        open_count = len(self.position_manager.get_all_open())
        if open_count >= self.config.max_concurrent_positions:
            return EntryRiskDecision(False, 0.0, "max_concurrent_positions")
        if edge <= 0:
            return EntryRiskDecision(False, 0.0, "non_positive_edge")
        if self._daily_loss_exceeded():
            self._activate_circuit_breaker("daily_loss_limit")
            return EntryRiskDecision(False, 0.0, "daily_loss_limit")
        if self._drawdown_exceeded():
            self._activate_circuit_breaker("max_drawdown")
            return EntryRiskDecision(False, 0.0, "max_drawdown")

        size = self._position_size(edge)
        if size < self.config.min_order_size_usdc:
            return EntryRiskDecision(False, 0.0, "size_below_min_order")
        return EntryRiskDecision(True, size, "allowed")

    def record_loss(self, realized_pnl: float) -> None:
        """Record realized PnL and activate circuit breakers when needed."""

        self._reset_if_new_day()
        self._realized_pnl += float(realized_pnl)
        self._trade_count += 1
        self._peak_equity = max(self._peak_equity, self._current_equity())
        if self._daily_loss_exceeded():
            self._activate_circuit_breaker("daily_loss_limit")
        elif self._drawdown_exceeded():
            self._activate_circuit_breaker("max_drawdown")

    def get_daily_stats(self) -> DailyRiskStats:
        """Return current UTC-day risk stats."""

        self._reset_if_new_day()
        equity = self._current_equity()
        drawdown = 0.0 if self._peak_equity <= 0 else max(0.0, (self._peak_equity - equity) / self._peak_equity)
        return DailyRiskStats(
            date=self._date,
            realized_pnl_usdc=self._realized_pnl,
            trade_count=self._trade_count,
            remaining_loss_budget_usdc=max(
                0.0,
                self.config.daily_loss_limit_usdc + self._realized_pnl,
            ),
            current_equity_usdc=equity,
            peak_equity_usdc=self._peak_equity,
            drawdown_pct=drawdown,
            circuit_breaker_active=self._circuit_breaker_reason is not None,
        )

    def reset_daily(self) -> None:
        """Reset UTC daily counters and reopen entry checks."""

        self._date = _utc_date()
        self._realized_pnl = 0.0
        self._trade_count = 0
        self._peak_equity = self.config.starting_equity_usdc
        self._circuit_breaker_reason = None

    def _position_size(self, edge: float) -> float:
        if edge >= self.config.min_edge_for_full_size:
            return self.config.max_position_size_usdc
        raw = self.config.kelly_fraction * edge * self.config.available_budget_usdc
        proportional_cap = (
            self.config.max_position_size_usdc
            * edge
            / self.config.min_edge_for_full_size
        )
        return min(raw, proportional_cap, self.config.max_position_size_usdc)

    def _current_equity(self) -> float:
        return self.config.starting_equity_usdc + self._realized_pnl

    def _daily_loss_exceeded(self) -> bool:
        return self._realized_pnl <= -self.config.daily_loss_limit_usdc

    def _drawdown_exceeded(self) -> bool:
        stats = self.get_daily_stats_without_reset()
        return stats.drawdown_pct >= self.config.max_drawdown_pct

    def get_daily_stats_without_reset(self) -> DailyRiskStats:
        equity = self._current_equity()
        drawdown = 0.0 if self._peak_equity <= 0 else max(0.0, (self._peak_equity - equity) / self._peak_equity)
        return DailyRiskStats(
            date=self._date,
            realized_pnl_usdc=self._realized_pnl,
            trade_count=self._trade_count,
            remaining_loss_budget_usdc=max(
                0.0,
                self.config.daily_loss_limit_usdc + self._realized_pnl,
            ),
            current_equity_usdc=equity,
            peak_equity_usdc=self._peak_equity,
            drawdown_pct=drawdown,
            circuit_breaker_active=self._circuit_breaker_reason is not None,
        )

    def _activate_circuit_breaker(self, reason: str) -> None:
        if self._circuit_breaker_reason is not None:
            return
        self._circuit_breaker_reason = reason
        if self._incident_conn is None:
            return
        stats = self.get_daily_stats_without_reset().to_dict()
        details = {"reason": reason, "stats": stats, "config": asdict(self.config)}
        ts = _now_ms()
        record_data_quality_incident(
            self._incident_conn,
            DataQualityIncident(
                incident_id=f"execution-risk:CIRCUIT_BREAKER:{reason}:{self._date}",
                source="execution_risk",
                incident_type="quality_rule_failure",
                severity="critical",
                started_at=ts,
                affected_symbol=None,
                details_json=json.dumps(details, sort_keys=True),
                alert_id="CIRCUIT_BREAKER",
                owner="ml-oncall",
            ),
            replace=True,
        )

    def _reset_if_new_day(self) -> None:
        if _utc_date() != self._date:
            self.reset_daily()


def _validate_config(config: RiskConfig) -> None:
    if config.max_position_size_usdc <= 0:
        raise ValueError("max_position_size_usdc must be positive")
    if config.max_concurrent_positions <= 0:
        raise ValueError("max_concurrent_positions must be positive")
    if config.daily_loss_limit_usdc <= 0:
        raise ValueError("daily_loss_limit_usdc must be positive")
    if not 0 < config.max_drawdown_pct <= 1:
        raise ValueError("max_drawdown_pct must be in (0, 1]")
    if not 0 < config.kelly_fraction <= 1:
        raise ValueError("kelly_fraction must be in (0, 1]")
    if config.min_edge_for_full_size <= 0:
        raise ValueError("min_edge_for_full_size must be positive")
    if config.available_budget_usdc <= 0:
        raise ValueError("available_budget_usdc must be positive")
    if config.min_order_size_usdc < 0:
        raise ValueError("min_order_size_usdc must be non-negative")
    if config.starting_equity_usdc <= 0:
        raise ValueError("starting_equity_usdc must be positive")


def _utc_date() -> str:
    return datetime.now(tz=UTC).date().isoformat()


def _now_ms() -> int:
    return int(time.time() * 1000)
