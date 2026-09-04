from __future__ import annotations

import subprocess
import sys

import pytest

from bigan.paper_trading.operator.ownership import AccountOwnershipError, AccountProcessLock

LOCK_SCRIPT = """
import sys
from pathlib import Path
from bigan.paper_trading.operator.ownership import AccountOwnershipError, AccountProcessLock
lock = AccountProcessLock(output_dir=Path(sys.argv[1]), operator_id=sys.argv[2], account_id='account')
try:
    lock.acquire()
except AccountOwnershipError:
    print('blocked', flush=True)
else:
    print('owned', flush=True)
    if sys.argv[3] == 'wait':
        sys.stdin.read(1)
    lock.release()
"""


@pytest.mark.parametrize("other_operator", ["operator", "different-operator"])
def test_account_is_exclusive_across_real_processes_and_operator_ids(tmp_path, other_operator):
    lock = AccountProcessLock(output_dir=tmp_path, operator_id="operator", account_id="account")
    lock.acquire()
    try:
        result = subprocess.run(
            [sys.executable, "-c", LOCK_SCRIPT, str(tmp_path), other_operator, "exit"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        assert result.stdout.strip() == "blocked"
    finally:
        lock.release()
    result = subprocess.run(
        [sys.executable, "-c", LOCK_SCRIPT, str(tmp_path), other_operator, "exit"],
        capture_output=True, text=True, check=True, timeout=10,
    )
    assert result.stdout.strip() == "owned"


def test_process_exit_releases_lock_without_deleting_its_inode(tmp_path):
    with subprocess.Popen(
        [sys.executable, "-c", LOCK_SCRIPT, str(tmp_path), "operator", "wait"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ) as child:
        assert child.stdout.readline().strip() == "owned"
        lock = AccountProcessLock(output_dir=tmp_path, operator_id="operator", account_id="account")
        with pytest.raises(AccountOwnershipError):
            lock.acquire()
        inode = lock.paths[0].stat().st_ino
        child.terminate()
        child.wait(timeout=10)
        lock.acquire()
        assert lock.paths[0].stat().st_ino == inode
        lock.release()
