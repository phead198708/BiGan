#!/usr/bin/env python3
"""Bridge local champion prediction events to a lightweight remote executor queue.

The local machine runs the expensive capture/feature/model pipeline. This bridge
tails local ``prediction_events`` and appends validated BTC-15M signal rows to a
remote JSONL file over SSH. The remote host can then run
``polymarket_phase4_live_champion_executor.py --signal-jsonl-path ...`` and keep
CPU use low because it only validates current CLOB liquidity and executes orders.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from bigan.monitoring.market_quality import (
    round_end_ts_from_canonical_symbol,
    tradable_market_implied_probability,
)


@dataclass(frozen=True, slots=True)
class BridgeSignal:
    event_id: str
    ts: int
    created_at: int
    model_version: str
    prob_up_15m: float
    canonical_symbol: str
    token_id: str
    outcome_side: str
    round_slug: str
    round_end_ts: int
    market_implied_prob: float
    token_probability: float
    edge: float
    bridged_at: int
    opposite_token_id: str = ""


def main() -> int:
    args = _parse_args()
    if args.poll_seconds <= 0:
        raise SystemExit("--poll-seconds must be positive")
    if args.batch_limit <= 0:
        raise SystemExit("--batch-limit must be positive")
    if args.start not in {"latest", "beginning"}:
        raise SystemExit("--start must be latest or beginning")
    allowed_families = frozenset(
        family.strip().upper() for family in args.market_families.split(",") if family.strip()
    )
    if not allowed_families:
        raise SystemExit("--market-families must list at least one family")

    cursor_created_at, cursor_event_id = (
        (0, "")
        if args.start == "beginning"
        else _latest_cursor(args.monitoring_db_path, args.model_version)
    )
    audit_log_path = Path(args.audit_log_path) if args.audit_log_path else None
    if audit_log_path is not None:
        audit_log_path.parent.mkdir(parents=True, exist_ok=True)

    print("bridging champion signals")
    print(f"local_db={args.monitoring_db_path}")
    print(f"model={args.model_version} start={args.start}")
    print(f"families={','.join(sorted(allowed_families))}")
    print(f"remote={args.remote} remote_path={args.remote_path}")
    print(f"cursor created_at={cursor_created_at} event_id={cursor_event_id}")

    while True:
        try:
            signals = _read_bridge_signals_after(
                args.monitoring_db_path,
                model_version=args.model_version,
                allowed_families=allowed_families,
                after_created_at=cursor_created_at,
                after_event_id=cursor_event_id,
                limit=args.batch_limit,
            )
        except Exception as exc:  # noqa: BLE001 - bridge should keep listening through locks.
            print(f"local db read error: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(args.poll_seconds)
            continue

        if not signals:
            print("waiting for local champion prediction events...", flush=True)
            time.sleep(args.poll_seconds)
            continue

        bridge_attempted_at = _now_ms()
        signals_to_send = [replace(signal, bridged_at=bridge_attempted_at) for signal in signals]
        payload = "".join(json.dumps(asdict(signal), sort_keys=True) + "\n" for signal in signals_to_send)

        result = _append_remote_jsonl(
            remote=args.remote,
            remote_path=args.remote_path,
            payload=payload,
            connect_timeout_seconds=args.ssh_connect_timeout_seconds,
            user_known_hosts_file=args.ssh_user_known_hosts_file,
            strict_host_key_checking=args.ssh_strict_host_key_checking,
        )
        if result.returncode != 0:
            print(
                "remote append failed; will retry without advancing cursor: "
                f"returncode={result.returncode} stderr={result.stderr.strip()}",
                flush=True,
            )
            time.sleep(args.poll_seconds)
            continue

        bridge_ack_at = _now_ms()
        if audit_log_path is not None:
            with audit_log_path.open("a", encoding="utf-8") as handle:
                handle.write(payload)
        cursor_created_at = signals[-1].created_at
        cursor_event_id = signals[-1].event_id
        max_signal_to_bridge_ack_ms = max(
            bridge_ack_at - signal.created_at for signal in signals if signal.created_at > 0
        )
        print(
            f"bridged {len(signals)} signal(s); "
            f"cursor created_at={cursor_created_at} event_id={cursor_event_id} "
            f"remote_append_ms={bridge_ack_at - bridge_attempted_at} "
            f"max_signal_to_bridge_ack_ms={max_signal_to_bridge_ack_ms}",
            flush=True,
        )
        if args.once:
            break
        time.sleep(args.poll_seconds)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monitoring-db-path", default="data/mlops/champion_catalog.duckdb")
    parser.add_argument("--model-version", default="xgboost-v4")
    parser.add_argument(
        "--market-families",
        default="BTC-15M",
        help=(
            "Comma-separated market families to bridge (e.g. BTC-15M,ETH-15M). "
            "Only signals whose canonical_symbol family is in this set are forwarded."
        ),
    )
    parser.add_argument(
        "--remote",
        required=True,
        help="SSH target for the execution host, for example ubuntu@13.231.238.96.",
    )
    parser.add_argument(
        "--remote-path",
        default="/home/ubuntu/BiGan/data/live/remote-signals/champion-signals.jsonl",
        help="Append-only JSONL queue path on the execution host.",
    )
    parser.add_argument(
        "--start",
        choices=("latest", "beginning"),
        default="latest",
        help="Use latest to ignore historical local predictions on startup.",
    )
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--batch-limit", type=int, default=100)
    parser.add_argument("--ssh-connect-timeout-seconds", type=int, default=10)
    parser.add_argument(
        "--ssh-user-known-hosts-file",
        default="",
        help="Optional known_hosts file for the bridge SSH command.",
    )
    parser.add_argument(
        "--ssh-strict-host-key-checking",
        default="accept-new",
        help="StrictHostKeyChecking value for the bridge SSH command.",
    )
    parser.add_argument("--audit-log-path", default="")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def _latest_cursor(db_path: str, model_version: str) -> tuple[int, str]:
    with duckdb.connect(db_path, read_only=True) as conn:
        row = conn.execute(
            """
            SELECT created_at, event_id
            FROM prediction_events
            WHERE model_version = ?
            ORDER BY created_at DESC, event_id DESC
            LIMIT 1
            """,
            [model_version],
        ).fetchone()
    if row is None:
        return 0, ""
    return int(row[0]), str(row[1])


def _read_bridge_signals_after(
    db_path: str,
    *,
    model_version: str,
    after_created_at: int,
    after_event_id: str,
    limit: int,
    allowed_families: frozenset[str] = frozenset({"BTC-15M"}),
) -> list[BridgeSignal]:
    family_clause = " OR ".join(
        "json_extract_string(feature_snapshot_json, '$.canonical_symbol') LIKE ?"
        for _ in allowed_families
    )
    family_params = [f"{family}:%" for family in sorted(allowed_families)]
    with duckdb.connect(db_path, read_only=True) as conn:
        rows = conn.execute(
            f"""
            SELECT event_id, ts, created_at, prob_up_15m, feature_snapshot_json
            FROM prediction_events
            WHERE model_version = ?
              AND ({family_clause})
              AND (
                    created_at > ?
                 OR (created_at = ? AND event_id > ?)
              )
            ORDER BY created_at ASC, event_id ASC
            LIMIT ?
            """,
            [model_version, *family_params, after_created_at, after_created_at, after_event_id, limit],
        ).fetchall()
        signals: list[BridgeSignal] = []
        for row in rows:
            signal = _bridge_signal_from_row(
                row, model_version=model_version, allowed_families=allowed_families
            )
            if signal is not None:
                family = signal.canonical_symbol.split(":", 1)[0]
                signals.append(
                    replace(
                        signal,
                        opposite_token_id=_opposite_token_id(
                            conn,
                            model_version=model_version,
                            family=family,
                            round_slug=signal.round_slug,
                            outcome_side=signal.outcome_side,
                        ),
                    )
                )
    return signals


def _bridge_signal_from_row(
    row: tuple[Any, ...],
    *,
    model_version: str,
    allowed_families: frozenset[str] = frozenset({"BTC-15M"}),
) -> BridgeSignal | None:
    event_id, ts, created_at, prob_up_15m, snapshot_json = row
    try:
        snapshot = json.loads(str(snapshot_json))
    except json.JSONDecodeError:
        return None
    if not isinstance(snapshot, dict):
        return None
    canonical_symbol = str(snapshot.get("canonical_symbol") or snapshot.get("symbol") or "")
    parts = canonical_symbol.split(":")
    if len(parts) < 3:
        return None
    family, round_slug, side = parts[0].upper(), parts[-2], parts[-1].upper()
    if family not in allowed_families or side not in {"UP", "DOWN"}:
        return None
    token_id = str(snapshot.get("source_symbol") or snapshot.get("token_id") or "")
    market = _market_implied_probability(snapshot, event_ts=int(ts))
    round_end_ts = round_end_ts_from_canonical_symbol(canonical_symbol)
    if not token_id or market is None or round_end_ts is None:
        return None
    prob = float(prob_up_15m)
    token_probability = 1.0 - prob if side == "DOWN" else prob
    return BridgeSignal(
        event_id=str(event_id),
        ts=int(ts),
        created_at=int(created_at),
        model_version=model_version,
        prob_up_15m=prob,
        canonical_symbol=canonical_symbol,
        token_id=token_id,
        outcome_side=side,
        round_slug=round_slug,
        round_end_ts=round_end_ts,
        market_implied_prob=market,
        token_probability=token_probability,
        edge=token_probability - market,
        bridged_at=_now_ms(),
    )


def _opposite_token_id(
    conn: duckdb.DuckDBPyConnection,
    *,
    model_version: str,
    family: str,
    round_slug: str,
    outcome_side: str,
) -> str:
    opposite_side = "DOWN" if outcome_side == "UP" else "UP"
    canonical_symbol = f"{family}:{round_slug}:{opposite_side}"
    row = conn.execute(
        """
        SELECT feature_snapshot_json
        FROM prediction_events
        WHERE model_version = ?
          AND json_extract_string(feature_snapshot_json, '$.canonical_symbol') = ?
        ORDER BY created_at DESC, event_id DESC
        LIMIT 1
        """,
        [model_version, canonical_symbol],
    ).fetchone()
    if row is None:
        return ""
    try:
        snapshot = json.loads(str(row[0]))
    except json.JSONDecodeError:
        return ""
    if not isinstance(snapshot, dict):
        return ""
    return str(snapshot.get("source_symbol") or snapshot.get("token_id") or "")


def _append_remote_jsonl(
    *,
    remote: str,
    remote_path: str,
    payload: str,
    connect_timeout_seconds: int,
    user_known_hosts_file: str = "",
    strict_host_key_checking: str = "accept-new",
) -> subprocess.CompletedProcess[str]:
    remote_file = shlex.quote(remote_path)
    remote_dir = shlex.quote(str(Path(remote_path).parent))
    command = f"mkdir -p {remote_dir} && cat >> {remote_file}"
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
    return subprocess.run(
        ssh_command,
        input=payload,
        text=True,
        capture_output=True,
        check=False,
    )


def _market_implied_probability(snapshot: dict[str, Any], *, event_ts: int) -> float | None:
    return tradable_market_implied_probability(snapshot, event_ts=event_ts)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=UTC).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
