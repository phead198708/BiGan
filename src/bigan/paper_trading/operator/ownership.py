"""Non-blocking POSIX process ownership for an operator and paper account."""

from __future__ import annotations

import fcntl
import hashlib
from pathlib import Path
from typing import BinaryIO


class AccountOwnershipError(RuntimeError):
    """Another process already owns this operator or account."""


class AccountProcessLock:
    def __init__(self, *, output_dir: Path, operator_id: str, account_id: str) -> None:
        self._handles: list[BinaryIO] = []
        account_key = hashlib.sha256(account_id.encode()).hexdigest()
        self.paths = (
            output_dir / ".account-locks" / f"{account_key}.lock",
            output_dir / operator_id / ".operator.lock",
        )

    @property
    def held(self) -> bool:
        return len(self._handles) == len(self.paths)

    def acquire(self) -> None:
        if self.held:
            return
        try:
            for path in self.paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                handle = path.open("a+b")
                self._handles.append(handle)
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise AccountOwnershipError("paper account/operator already has a writer") from exc
        except BaseException:
            self.release()
            raise

    def release(self) -> None:
        # Do not unlink lock files: replacing their inodes defeats exclusion.
        for handle in reversed(self._handles):
            handle.close()
        self._handles.clear()

    def __del__(self) -> None:
        # A final safeguard for abandoned embedding objects; normal operation
        # releases explicitly in operator.shutdown(), process exit also closes FDs.
        self.release()
