#!/usr/bin/env python3
"""Mirror a local current-round signal JSONL queue to a remote host over SSH."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


def main() -> int:
    args = _parse_args()
    if args.poll_seconds <= 0:
        raise SystemExit("--poll-seconds must be positive")
    local_path = Path(args.local_path)
    last_signature = ""
    print("mirroring signal jsonl queue")
    print(f"local_path={local_path}")
    print(f"remote={args.remote} remote_path={args.remote_path}")
    while True:
        payload = _read_complete_jsonl_bytes(local_path)
        signature = hashlib.sha256(payload).hexdigest()
        if signature != last_signature:
            result = _replace_remote_file(
                remote=args.remote,
                remote_path=args.remote_path,
                payload=payload,
                connect_timeout_seconds=args.ssh_connect_timeout_seconds,
                command_timeout_seconds=args.ssh_command_timeout_seconds,
                user_known_hosts_file=args.ssh_user_known_hosts_file,
                strict_host_key_checking=args.ssh_strict_host_key_checking,
            )
            if result.returncode != 0:
                print(
                    "remote replace failed; will retry: "
                    f"returncode={result.returncode} stderr={result.stderr.strip()}",
                    flush=True,
                )
                time.sleep(args.poll_seconds)
                continue
            last_signature = signature
            print(
                "mirrored "
                f"bytes={len(payload)} lines={_line_count(payload)} "
                f"round={_last_round_slug(payload)} signature={signature[:12]}",
                flush=True,
            )
            if args.once:
                break
        elif args.once:
            break
        time.sleep(args.poll_seconds)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-path", required=True)
    parser.add_argument("--remote", required=True)
    parser.add_argument("--remote-path", required=True)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--ssh-connect-timeout-seconds", type=int, default=10)
    parser.add_argument("--ssh-command-timeout-seconds", type=int, default=20)
    parser.add_argument("--ssh-user-known-hosts-file", default="")
    parser.add_argument("--ssh-strict-host-key-checking", default="accept-new")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def _read_complete_jsonl_bytes(path: Path) -> bytes:
    if not path.exists():
        return b""
    payload = path.read_bytes()
    if not payload or payload.endswith(b"\n"):
        return payload
    last_newline = payload.rfind(b"\n")
    if last_newline < 0:
        return b""
    return payload[: last_newline + 1]


def _replace_remote_file(
    *,
    remote: str,
    remote_path: str,
    payload: bytes,
    connect_timeout_seconds: int,
    command_timeout_seconds: int,
    user_known_hosts_file: str = "",
    strict_host_key_checking: str = "accept-new",
) -> subprocess.CompletedProcess[bytes]:
    remote_file = shlex.quote(remote_path)
    remote_dir = shlex.quote(str(Path(remote_path).parent))
    command = (
        f"mkdir -p {remote_dir} && "
        f"tmp=$(mktemp {remote_dir}/.mirror.XXXXXX) && "
        f"cat > \"$tmp\" && "
        f"mv \"$tmp\" {remote_file}"
    )
    ssh_command = [
        "ssh",
        "-o",
        f"ConnectTimeout={connect_timeout_seconds}",
        "-o",
        "BatchMode=yes",
        "-o",
        f"StrictHostKeyChecking={strict_host_key_checking}",
    ]
    if user_known_hosts_file:
        ssh_command.extend(["-o", f"UserKnownHostsFile={user_known_hosts_file}"])
    ssh_command.extend([remote, command])
    try:
        return subprocess.run(
            ssh_command,
            input=payload,
            capture_output=True,
            check=False,
            timeout=command_timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stderr = (exc.stderr or b"") + (
            f"ssh command timed out after {command_timeout_seconds}s".encode()
        )
        return subprocess.CompletedProcess(ssh_command, 124, exc.output or b"", stderr)


def _line_count(payload: bytes) -> int:
    if not payload:
        return 0
    return payload.count(b"\n")


def _last_round_slug(payload: bytes) -> str:
    for raw in reversed(payload.splitlines()):
        if not raw.strip():
            continue
        try:
            item: Any = json.loads(raw)
        except json.JSONDecodeError:
            return "<invalid-json>"
        if isinstance(item, dict):
            return str(item.get("round_slug") or "")
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
