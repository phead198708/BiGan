#!/usr/bin/env python3
"""Supervise runbook-compliant xgboost-v5 Phase 4 windows.

The executor keeps the locked trading caps (`max_rounds=6`, 1 USDC, one
concurrent position) while staying alive for the full stage runtime. This script
chains multiple isolated signal queues/log directories and stops if a lifecycle
gate is not clean.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import pathlib
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class Stage:
    number: int
    run_id: str
    ts: str
    local_bridge_screen: str
    local_bridge_log: str
    remote_executor_screen: str
    remote_queue: str
    remote_log_dir: str


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _utc_stamp() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _log(message: str) -> None:
    print(f"[{_utc_now().strftime('%Y-%m-%dT%H:%M:%SZ')}] {message}", flush=True)


def _run(args: list[str], *, timeout: float = 30, check: bool = False) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(args, text=True, capture_output=True, timeout=timeout)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"command failed rc={proc.returncode} args={args!r} "
            f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )
    return proc


def _ssh(remote: str, command: str, *, timeout: float = 30, check: bool = False) -> subprocess.CompletedProcess[str]:
    return _run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", remote, command],
        timeout=timeout,
        check=check,
    )


def _summary_path(remote: str, stage: Stage) -> str:
    try:
        out = _ssh(
            remote,
            f"ls -1 {shlex.quote(stage.remote_log_dir)}/*-summary.json 2>/dev/null | head -1 || true",
            timeout=30,
        ).stdout.strip()
    except subprocess.TimeoutExpired:
        _log(f"summary probe timed out for {stage.run_id}; will retry")
        return ""
    return out.splitlines()[0] if out else ""


def _read_summary(remote: str, path: str) -> dict[str, Any]:
    return json.loads(_ssh(remote, f"cat {shlex.quote(path)}", timeout=30, check=True).stdout)


def _last_remote_event(remote: str, stage: Stage) -> str:
    try:
        return _ssh(
            remote,
            f"tail -1 {shlex.quote(stage.remote_log_dir)}/phase4-*.jsonl 2>/dev/null || true",
            timeout=30,
        ).stdout.strip()
    except subprocess.TimeoutExpired:
        _log(f"last-event probe timed out for {stage.run_id}; will retry")
        return ""


def _queue_lines(remote: str, stage: Stage) -> str:
    try:
        return _ssh(
            remote,
            f"wc -l < {shlex.quote(stage.remote_queue)} 2>/dev/null || echo missing",
            timeout=30,
        ).stdout.strip()
    except subprocess.TimeoutExpired:
        _log(f"queue probe timed out for {stage.run_id}; will retry")
        return "timeout"


def _stage_config_from_summary(remote: str, summary: dict[str, Any]) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    summary_cfg = summary.get("config")
    if isinstance(summary_cfg, dict):
        cfg.update(summary_cfg)
    for key in (
        "model_version",
        "market_families",
        "edge_threshold",
        "settlement_edge_threshold",
        "volatility_score_threshold",
        "volatility_min_entry_price",
        "volatility_min_seconds_to_expiry",
        "enable_volatility_live_entries",
        "max_rounds",
        "max_runtime_minutes",
        "max_position_size_usdc",
        "max_concurrent_positions",
        "min_entry_price",
        "continue_after_max_rounds_until_runtime",
    ):
        if key in summary and summary[key] is not None:
            cfg[key] = summary[key]

    log_path = summary.get("execution_log_path")
    if not log_path:
        return cfg

    started_line = _ssh(
        remote,
        f"grep -m 1 '\"phase4_started\"' {shlex.quote(str(log_path))} 2>/dev/null || true",
        timeout=30,
    ).stdout.strip()
    if not started_line:
        return cfg
    try:
        started = json.loads(started_line)
    except json.JSONDecodeError:
        return cfg
    started_cfg = started.get("config")
    if isinstance(started_cfg, dict):
        started_cfg.update(cfg)
        return started_cfg
    return cfg


def _clean_blockers(
    summary: dict[str, Any],
    *,
    require_window_flag: bool,
    config: dict[str, Any] | None = None,
) -> list[str]:
    blockers: list[str] = []
    if summary.get("status") not in {None, "LIFECYCLE_PASS"}:
        blockers.append(f"status={summary.get('status')!r}")
    for key in ("open_positions_at_shutdown", "exits_pending_confirmation", "exits_pending_settlement"):
        if int(summary.get(key) or 0) != 0:
            blockers.append(f"{key}={summary.get(key)}")
    skipped = summary.get("skipped") or {}
    if int(skipped.get("daily_loss_stop") or 0) != 0:
        blockers.append(f"daily_loss_stop={skipped.get('daily_loss_stop')}")
    cfg = config or summary.get("config") or {}
    checks: dict[str, Any] = {
        "model_version": "xgboost-v5",
        "edge_threshold": 0.45,
        "settlement_edge_threshold": 0.45,
        "volatility_score_threshold": 0.50,
        "volatility_min_entry_price": 0.20,
        "volatility_min_seconds_to_expiry": 420.0,
        "enable_volatility_live_entries": False,
        "max_rounds": 6,
        "max_runtime_minutes": 120.0,
        "max_position_size_usdc": 1.0,
        "max_concurrent_positions": 1,
        "min_entry_price": 0.35,
    }
    for key, expected in checks.items():
        got = cfg.get(key)
        if isinstance(expected, float):
            ok = got is not None and abs(float(got) - expected) < 1e-9
        else:
            ok = got == expected
        if not ok:
            blockers.append(f"config.{key}={got!r}")
    if sorted(cfg.get("market_families") or []) != ["BTC-15M", "ETH-15M"]:
        blockers.append(f"config.market_families={cfg.get('market_families')!r}")
    if require_window_flag and cfg.get("continue_after_max_rounds_until_runtime") is not True:
        blockers.append(
            f"config.continue_after_max_rounds_until_runtime={cfg.get('continue_after_max_rounds_until_runtime')!r}"
        )
    return blockers


def _stop_local_bridge(stage: Stage) -> None:
    if stage.local_bridge_screen:
        _run(["screen", "-S", stage.local_bridge_screen, "-X", "quit"], timeout=10)
    try:
        ps = _run(["ps", "-axo", "pid=,command="], timeout=10, check=True).stdout
    except Exception as exc:  # pragma: no cover - defensive operational cleanup
        _log(f"could not inspect local processes for bridge cleanup: {exc}")
        return
    victims: list[int] = []
    for line in ps.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_s, _, command = line.partition(" ")
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        if "scripts/champion_signal_bridge.py" in command and stage.remote_queue in command:
            victims.append(pid)
    for pid in victims:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGTERM)
    if victims:
        _log(f"stopped local bridge processes for {stage.run_id}: {victims}")


def _start_stage(args: argparse.Namespace, stage_no: int) -> Stage:
    ts = _utc_stamp()
    run_id = f"{ts}-stage{stage_no}"
    remote_queue = (
        f"/home/ubuntu/BiGan/data/live/remote-signals/"
        f"champion-signals-runbook-12h-{run_id}.jsonl"
    )
    remote_log_dir = f"/home/ubuntu/BiGan/logs/phase4-v5-runbook-12h-{run_id}"
    remote_screen = f"phase4_executor_runbook_12h_{ts}_s{stage_no}"
    local_screen = f"phase4_v5_runbook_12h_bridge_{ts}_s{stage_no}"
    local_log_dir = pathlib.Path(args.local_cwd) / "data" / "logs" / f"phase4-v5-runbook-12h-{run_id}"
    local_log_dir.mkdir(parents=True, exist_ok=True)
    local_bridge_log = f"data/logs/phase4-v5-runbook-12h-{run_id}/bridge.stdout.log"

    executor_cmd = (
        "cd /home/ubuntu/BiGan && "
        "set -a && . ~/.config/bigan/polymarket.env && set +a && "
        "CONFIRM=yes "
        "MODEL_VERSION=xgboost-v5 "
        "MARKET_FAMILIES=BTC-15M,ETH-15M "
        "SETTLEMENT_EDGE_THRESHOLD=0.45 "
        "VOLATILITY_SCORE_THRESHOLD=0.50 "
        "VOLATILITY_MIN_ENTRY_PRICE=0.20 "
        "MAX_POSITION_SIZE_USDC=1.0 "
        "MAX_CONCURRENT_POSITIONS=1 "
        "MAX_ROUNDS=6 "
        "DAILY_LOSS_LIMIT_USDC=3.0 "
        "MAX_RUNTIME_MINUTES=120 "
        "MIN_ENTRY_PRICE=0.35 "
        "MIN_SECONDS_TO_EXPIRY=180 "
        "MAX_SECONDS_TO_EXPIRY=1200 "
        "POLL_SECONDS=10 "
        "CONTINUE_AFTER_MAX_ROUNDS_UNTIL_RUNTIME=true "
        f"SIGNAL_JSONL_PATH={shlex.quote(remote_queue)} "
        "SIGNAL_JSONL_START=tail "
        f"LOG_DIR={shlex.quote(remote_log_dir)} "
        f"./scripts/run_xgboost_v5_capped_live_shadow.sh > {shlex.quote(remote_log_dir)}/executor.stdout.log 2>&1"
    )
    remote_cmd = (
        "cd /home/ubuntu/BiGan && "
        f"mkdir -p /home/ubuntu/BiGan/data/live/remote-signals {shlex.quote(remote_log_dir)} && "
        f": > {shlex.quote(remote_queue)} && "
        f"screen -dmS {shlex.quote(remote_screen)} bash -lc {shlex.quote(executor_cmd)}"
    )
    _ssh(args.remote, remote_cmd, timeout=30, check=True)

    bridge_cmd = (
        f"cd {shlex.quote(args.local_cwd)} && "
        ".venv/bin/python scripts/champion_signal_bridge.py "
        "--monitoring-db-path data/mlops/champion_catalog.duckdb "
        "--model-version xgboost-v5 "
        "--market-families BTC-15M,ETH-15M "
        "--outcome-side UP "
        f"--remote {shlex.quote(args.remote)} "
        f"--remote-path {shlex.quote(remote_queue)} "
        f"--start latest > {shlex.quote(local_bridge_log)} 2>&1"
    )
    _run(["screen", "-dmS", local_screen, "bash", "-lc", bridge_cmd], timeout=10, check=True)
    return Stage(
        number=stage_no,
        run_id=run_id,
        ts=ts,
        local_bridge_screen=local_screen,
        local_bridge_log=local_bridge_log,
        remote_executor_screen=remote_screen,
        remote_queue=remote_queue,
        remote_log_dir=remote_log_dir,
    )


def _wait_for_stage_summary(
    args: argparse.Namespace,
    stage: Stage,
    *,
    require_window_flag: bool,
    label: str,
) -> dict[str, Any]:
    while True:
        path = _summary_path(args.remote, stage)
        if path:
            summary = _read_summary(args.remote, path)
            _log(
                f"{label} summary path={path} status={summary.get('status')} "
                f"open={summary.get('open_positions_at_shutdown')} "
                f"filled={summary.get('entries_filled')} closed={summary.get('closes_filled')}"
            )
            config = _stage_config_from_summary(args.remote, summary)
            blockers = _clean_blockers(summary, require_window_flag=require_window_flag, config=config)
            _stop_local_bridge(stage)
            if blockers:
                raise RuntimeError(f"{label} not clean: {', '.join(blockers)}")
            return summary

        q = _queue_lines(args.remote, stage)
        last = _last_remote_event(args.remote, stage)
        _log(f"{label} active queue_lines={q} last={last[:240]}")
        time.sleep(args.check_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default="ubuntu@54.250.242.139")
    parser.add_argument("--local-cwd", default="/Users/tcscoder/Workspaces/BiGan")
    parser.add_argument("--total-stages", type=int, default=6)
    parser.add_argument("--start-stage", type=int, default=1)
    parser.add_argument("--check-seconds", type=float, default=60.0)
    parser.add_argument("--preflight-run-id")
    parser.add_argument("--preflight-stage", type=int, default=0)
    parser.add_argument("--preflight-ts")
    parser.add_argument("--preflight-local-bridge-screen", default="")
    parser.add_argument("--preflight-local-bridge-log", default="")
    parser.add_argument("--preflight-remote-executor-screen", default="")
    parser.add_argument("--preflight-remote-queue", default="")
    parser.add_argument("--preflight-remote-log-dir", default="")
    parser.add_argument("--resume-run-id")
    parser.add_argument("--resume-stage", type=int, default=0)
    parser.add_argument("--resume-ts")
    parser.add_argument("--resume-local-bridge-screen", default="")
    parser.add_argument("--resume-local-bridge-log", default="")
    parser.add_argument("--resume-remote-executor-screen", default="")
    parser.add_argument("--resume-remote-queue", default="")
    parser.add_argument("--resume-remote-log-dir", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _log("12h window supervisor starting")

    if args.preflight_run_id:
        preflight = Stage(
            number=args.preflight_stage,
            run_id=args.preflight_run_id,
            ts=args.preflight_ts or "",
            local_bridge_screen=args.preflight_local_bridge_screen,
            local_bridge_log=args.preflight_local_bridge_log,
            remote_executor_screen=args.preflight_remote_executor_screen,
            remote_queue=args.preflight_remote_queue,
            remote_log_dir=args.preflight_remote_log_dir,
        )
        _wait_for_stage_summary(args, preflight, require_window_flag=False, label=f"preflight {preflight.run_id}")
        if args.start_stage <= 1:
            _log("preflight clean; starting fresh 6x2h runbook campaign")
        else:
            _log(f"preflight clean; continuing runbook campaign at stage {args.start_stage}/{args.total_stages}")

    start_stage = args.start_stage
    if args.resume_run_id:
        resume = Stage(
            number=args.resume_stage,
            run_id=args.resume_run_id,
            ts=args.resume_ts or "",
            local_bridge_screen=args.resume_local_bridge_screen,
            local_bridge_log=args.resume_local_bridge_log,
            remote_executor_screen=args.resume_remote_executor_screen,
            remote_queue=args.resume_remote_queue,
            remote_log_dir=args.resume_remote_log_dir,
        )
        _log(f"resuming active 12h stage {resume.number}/{args.total_stages}: {resume}")
        _wait_for_stage_summary(
            args,
            resume,
            require_window_flag=True,
            label=f"12h stage {resume.number}/{args.total_stages}",
        )
        start_stage = resume.number + 1

    for stage_no in range(start_stage, args.total_stages + 1):
        stage = _start_stage(args, stage_no)
        _log(f"started 12h stage {stage_no}/{args.total_stages}: {stage}")
        _wait_for_stage_summary(args, stage, require_window_flag=True, label=f"12h stage {stage_no}/{args.total_stages}")

    _log("all 12h stages complete; run account cash-flow reconciliation next")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
