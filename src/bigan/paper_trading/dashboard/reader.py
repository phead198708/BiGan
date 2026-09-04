"""Request-scoped, bounded reads. No operator/session/recovery lifecycle here."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bigan.paper_trading.operator.checkpoint import (
    AccountCheckpoint,
    AccountCheckpointStore,
    load_run_link,
)
from bigan.paper_trading.operator.config import OperatorConfig
from bigan.paper_trading.operator.read_model import (
    OperatorReadRepository,
    OperatorStatus,
    account_totals,
)
from bigan.paper_trading.storage import PaperRunStore

RUN_ID_PATTERN = re.compile(r"paper-[0-9a-f]{24}")
SECTIONS = ("runs", "decisions", "fills", "settlements")


class DashboardUnavailable(RuntimeError):
    """Sanitized availability failure, safe to expose over HTTP."""


def validate_cursor(value: str | None) -> None:
    if value is not None and not RUN_ID_PATTERN.fullmatch(value):
        raise ValueError("Invalid run cursor")


def _warning(code: str, message: str, section: str) -> dict[str, str]:
    return {"code": code, "message": message, "section": section}


class DashboardReader:
    """Resolve a fresh account frontier for every request, never cache a run.

    Consistency retries: at most three attempts and a 250 ms retry budget, with
    no sleeps. Filesystem operations themselves are not interruptible. Status
    and ledger observations within one run may have different event sequences;
    this is an observational view, not a transactional audit/recovery service.
    """

    def __init__(self, config: OperatorConfig, *, clock_ms: Callable[[], int] | None = None) -> None:
        self.config = config
        self.clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self.root = Path(config.output_dir).expanduser().resolve()
        self.status_path = self.root / config.operator_id / config.status_filename
        self.checkpoints = AccountCheckpointStore(self.root / config.operator_id / "account_checkpoint.json")
        self.stale_after_ms = max(3 * config.status_interval_ms, 5_000)

    def read_status(self) -> dict[str, Any]:
        try:
            status = self._status()
        except Exception:
            raise DashboardUnavailable("Operator status is unavailable") from None
        return self._base(status)

    def read(self, *, limit: int | None = None, before_run_id: str | None = None) -> dict[str, Any]:
        limit = self.config.recent_query_default if limit is None else limit
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= self.config.recent_query_max:
            raise ValueError("Limit is outside configured bounds")
        validate_cursor(before_run_id)
        deadline = time.monotonic() + 0.250
        status: OperatorStatus | None = None
        for _attempt in range(3):
            try:
                status = self._status()
            except Exception:
                raise DashboardUnavailable("Operator status is unavailable") from None
            view = self._base(status)
            try:
                checkpoint = self.checkpoints.load(config_sha256=self.config.config_sha256)
                if checkpoint is None and status.run_id is None:
                    view["warnings"].append(_warning("NO_ACTIVE_RUN", "No active run is available yet", "account"))
                    return view
                if checkpoint is None or checkpoint.activation_state != "ACTIVE" or checkpoint.run_id != status.run_id:
                    raise ValueError("account frontier is changing or unavailable")
                repository = self._repository(status, checkpoint)
                self._account(view, repository, checkpoint)
                for section in SECTIONS:
                    self._history(view, repository, section, limit, before_run_id)
                # Re-read atomic frontier and status after ALL section reads.
                # Never return a mix assembled across a rollover publication.
                after = self._status()
                if (
                    after.run_id != status.run_id
                    or after.process_started_at_ms != status.process_started_at_ms
                    or self.checkpoints.load(config_sha256=self.config.config_sha256) != checkpoint
                ):
                    raise ValueError("account frontier changed during read")
                view["active_market"] = {**(status.active_market or {}), "title": checkpoint.market.title}
                return view
            except Exception:
                if time.monotonic() >= deadline:
                    break
        # A currently readable status is useful even while activation is incomplete.
        view = self.read_status()
        view["warnings"].append(_warning(
            "FRONTIER_UNAVAILABLE", "Account and history are unavailable while the run frontier is inconsistent", "account",
        ))
        return view

    def _status(self) -> OperatorStatus:
        status = OperatorReadRepository(status_path=self.status_path, run_store=None).current_status()
        if (status.operator_id, status.strategy_id, status.source_commit) != (
            self.config.operator_id, self.config.strategy_id, self.config.source_commit,
        ):
            raise ValueError("operator status identity mismatch")
        for timestamp in (status.updated_at_ms, status.process_started_at_ms):
            if isinstance(timestamp, bool) or not isinstance(timestamp, int):
                raise ValueError("invalid status timestamp")
        expected_safety = {
            "capital_at_risk", "broker_exchange_write_enabled", "live_exchange_write_enabled",
            "polymarket_write_enabled", "wallet_signing_enabled",
        }
        if set(status.safety) != expected_safety or any(value is not False for value in status.safety.values()):
            raise ValueError("invalid operator safety boundary")
        validate_cursor(status.run_id)
        return status

    def _base(self, status: OperatorStatus) -> dict[str, Any]:
        now = self.clock_ms()
        age = now - status.updated_at_ms
        warnings = []
        stale = age < 0 or age > self.stale_after_ms
        if stale:
            warnings.append(_warning("STATUS_STALE", "Operator status is stale or its clock is ahead", "status"))
        return {
            "schema_version": 1, "generated_at_ms": now, "stale": stale,
            "operator_identity": self.operator_identity,
            "status_age_ms": age, "stale_after_ms": self.stale_after_ms,
            "status": status.to_dict(), "active_market": status.active_market,
            "account": None, "positions": None,
            "recent": dict.fromkeys(SECTIONS), "warnings": warnings,
            "query_defaults": {"limit": self.config.recent_query_default, "max_limit": self.config.recent_query_max},
        }

    @property
    def operator_identity(self) -> dict[str, str]:
        """Safe deployment identity; never expose the complete configuration."""
        return {
            "operator_id": self.config.operator_id,
            "strategy_id": self.config.strategy_id,
            "paper_account_id": self.config.paper_account_id,
            "source_commit": self.config.source_commit,
            "config_sha256": self.config.config_sha256,
        }

    def _repository(self, status: OperatorStatus, checkpoint: AccountCheckpoint) -> OperatorReadRepository:
        validate_cursor(checkpoint.run_id)
        if load_run_link(self.root, checkpoint.run_id, self.config.config_sha256) != checkpoint:
            raise ValueError("run link does not match checkpoint")
        store = PaperRunStore.open_read_only(output_dir=self.root, run_id=checkpoint.run_id)
        manifest, market = store.manifest, checkpoint.market
        if (
            manifest.run_id != status.run_id or manifest.initial_bankroll != checkpoint.opening_cash
            or manifest.source_commit != self.config.source_commit or manifest.fee_bps != self.config.fee_bps
            or manifest.window_ids != (market.window_id,) or manifest.market_symbols != (self.config.underlying,)
            or manifest.windows[0].start_ts_ms != market.start_ts_ms
            or manifest.windows[0].end_ts_ms != market.end_ts_ms
            or not re.fullmatch(r"[0-9a-f]{64}", manifest.config_sha256)
            or status.active_market is None or status.active_market.get("window_id") != market.window_id
            or status.active_market.get("market_id") != market.market_id
        ):
            raise ValueError("manifest/status/checkpoint identity mismatch")
        return OperatorReadRepository(
            status_path=self.status_path, run_store=store, checkpoint=checkpoint,
            default_limit=self.config.recent_query_default, max_limit=self.config.recent_query_max,
        )

    @staticmethod
    def _account(view: dict[str, Any], repository: OperatorReadRepository, checkpoint: AccountCheckpoint) -> None:
        try:
            snapshot = repository.current_account()
            if snapshot is None:
                raise ValueError("account is missing")
            view["account"] = {
                **account_totals(snapshot, checkpoint, checkpoint.opening_cash),
                "run_id": snapshot.run_id, "timestamp_ms": snapshot.timestamp_ms,
                "last_event_sequence": snapshot.last_event_sequence,
                "drawdown": snapshot.drawdown, "drawdown_scope": "current_run",
            }
            # Only display fields from the canonical ledger. Opened time is the
            # earliest remaining lot timestamp, not a new position/PnL model.
            view["positions"] = [
                {**position.to_dict(), "opened_at_ms": min(
                    (lot.entry_ts_ms for lot in snapshot.open_lots
                     if (lot.window_id, lot.side) == (position.window_id, position.side)), default=None,
                )}
                for position in snapshot.positions
            ]
        except Exception:
            view["warnings"].append(_warning("ACCOUNT_UNAVAILABLE", "Account snapshot is temporarily unavailable", "account"))

    @staticmethod
    def _history(view: dict[str, Any], repository: OperatorReadRepository, section: str,
                 limit: int, cursor: str | None) -> None:
        method = {"runs": "recent_runs", "decisions": "recent_decisions", "fills": "recent_fills", "settlements": "settlements"}[section]
        try:
            view["recent"][section] = list(getattr(repository, method)(limit, before_run_id=cursor))
        except Exception:
            view["warnings"].append(_warning("HISTORY_UNAVAILABLE", f"Recent {section} are temporarily unavailable", section))
