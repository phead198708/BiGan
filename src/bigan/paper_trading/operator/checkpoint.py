"""Durable account frontier, written before creating each successor run.

The run ledger remains authoritative for cash and settlement. This small
checkpoint names which ledger must be recovered, even if discovery has moved
on. An activation intent also survives a crash before the new run is created.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from .discovery import DiscoveredMarket


@dataclass(frozen=True, slots=True)
class AccountCheckpoint:
    config_sha256: str
    run_id: str
    market: DiscoveredMarket
    opening_cash: float
    initial_bankroll: float
    activation_state: Literal["ACTIVATING", "ACTIVE"] = "ACTIVATING"
    run_index: int = 0
    prior_realized_pnl: float = 0.0
    prior_fees: float = 0.0
    predecessor_run_id: str | None = None
    predecessor_window_id: str | None = None
    predecessor_settled_cash: float | None = None
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("unsupported account checkpoint schema")
        if self.activation_state not in {"ACTIVATING", "ACTIVE"}:
            raise ValueError("invalid account activation state")
        if not isinstance(self.run_index, int) or isinstance(self.run_index, bool) or self.run_index < 0:
            raise ValueError("invalid account run index")
        if not math.isfinite(self.initial_bankroll) or self.initial_bankroll <= 0:
            raise ValueError("account original bankroll must be positive")
        if not math.isfinite(self.prior_realized_pnl) or not math.isfinite(self.prior_fees) or self.prior_fees < 0:
            raise ValueError("invalid cumulative account totals")
        if not math.isfinite(self.opening_cash) or self.opening_cash <= 0:
            raise ValueError("account checkpoint requires positive opening cash")
        predecessor = (self.predecessor_run_id, self.predecessor_window_id,
                       self.predecessor_settled_cash)
        if any(value is not None for value in predecessor):
            if any(value is None for value in predecessor):
                raise ValueError("incomplete predecessor settlement identity")
            if self.predecessor_settled_cash != self.opening_cash:
                raise ValueError("successor cash must equal predecessor settlement cash")
            if self.run_index == 0:
                raise ValueError("successor must advance the account run index")
        elif (
            self.run_index != 0 or self.prior_fees != 0 or self.prior_realized_pnl != 0
            or self.opening_cash != self.initial_bankroll
        ):
            raise ValueError("first run must start at the original account bankroll")


class AccountCheckpointStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self, *, config_sha256: str) -> AccountCheckpoint | None:
        if not self.path.exists():
            return None
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != 2:
            raise ValueError("unsupported account checkpoint schema; explicit migration required")
        if set(payload) != set(AccountCheckpoint.__dataclass_fields__):
            raise ValueError("account checkpoint fields do not match schema")
        payload["market"] = DiscoveredMarket(**payload["market"])
        checkpoint = AccountCheckpoint(**payload)
        if checkpoint.config_sha256 != config_sha256:
            raise ValueError("account checkpoint configuration identity mismatch")
        return checkpoint

    def write(self, checkpoint: AccountCheckpoint) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(asdict(checkpoint), sort_keys=True, allow_nan=False) + "\n"
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent,
                prefix=".account-checkpoint-", delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            directory_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


RUN_LINK_FILE = "operator_account_link.json"


def run_link_store(output_dir: Path, run_id: str) -> AccountCheckpointStore:
    if run_id in {".", ".."} or Path(run_id).name != run_id:
        raise ValueError("invalid account run ID")
    return AccountCheckpointStore(output_dir / run_id / RUN_LINK_FILE)


def load_run_link(output_dir: Path, run_id: str, config_sha256: str) -> AccountCheckpoint:
    link = run_link_store(output_dir, run_id).load(config_sha256=config_sha256)
    if link is None or link.run_id != run_id or link.activation_state != "ACTIVE":
        raise ValueError("missing or invalid active account run link")
    return link


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
