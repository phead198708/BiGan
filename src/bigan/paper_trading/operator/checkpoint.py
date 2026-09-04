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

from .discovery import DiscoveredMarket


@dataclass(frozen=True, slots=True)
class AccountCheckpoint:
    config_sha256: str
    run_id: str
    market: DiscoveredMarket
    opening_cash: float
    predecessor_run_id: str | None = None
    predecessor_window_id: str | None = None
    predecessor_settled_cash: float | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported account checkpoint schema")
        if not math.isfinite(self.opening_cash) or self.opening_cash <= 0:
            raise ValueError("account checkpoint requires positive opening cash")
        predecessor = (self.predecessor_run_id, self.predecessor_window_id,
                       self.predecessor_settled_cash)
        if any(value is not None for value in predecessor):
            if any(value is None for value in predecessor):
                raise ValueError("incomplete predecessor settlement identity")
            if self.predecessor_settled_cash != self.opening_cash:
                raise ValueError("successor cash must equal predecessor settlement cash")


class AccountCheckpointStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self, *, config_sha256: str) -> AccountCheckpoint | None:
        if not self.path.exists():
            return None
        payload = json.loads(self.path.read_text(encoding="utf-8"))
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
