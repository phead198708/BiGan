#!/usr/bin/env python3
"""Monitor BTC-15m health and enforce transfer/training readiness gates."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.challenge_development_governance import (  # noqa: E402
    build_lane_health_summary,
    build_training_readiness,
    run_transfer_diagnostic_if_ready,
)
from bigan.v8.polymarket.challenge_development_lane import (  # noqa: E402
    SAFETY,
    atomic_write_json,
)

DEFAULT_TRANSFER_PROTOCOL = ROOT / (
    "examples/v8/polymarket_configs/challenge_model_15m_transfer_diagnostic_protocol.json"
)
DEFAULT_TRAINING_PROTOCOL = ROOT / (
    "examples/v8/polymarket_configs/challenge_model_15m_training_protocol_preregistration.json"
)


def run_service(
    *,
    service_root: Path | str,
    transfer_protocol_path: Path | str,
    transfer_protocol_sha256: str,
    training_protocol_path: Path | str,
    training_protocol_sha256: str,
    poll_seconds: float,
    run_once: bool,
) -> dict:
    root = Path(service_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "development_lane_governance_monitor.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("development lane governance monitor is already running") from exc
        while True:
            state = _cycle(
                root=root,
                transfer_protocol_path=Path(transfer_protocol_path).resolve(),
                transfer_protocol_sha256=transfer_protocol_sha256,
                training_protocol_path=Path(training_protocol_path).resolve(),
                training_protocol_sha256=training_protocol_sha256,
            )
            if run_once:
                return state
            time.sleep(poll_seconds)


def _cycle(
    *,
    root: Path,
    transfer_protocol_path: Path,
    transfer_protocol_sha256: str,
    training_protocol_path: Path,
    training_protocol_sha256: str,
) -> dict:
    health = build_lane_health_summary(lane_root=root)
    transfer = run_transfer_diagnostic_if_ready(
        lane_root=root,
        protocol_path=transfer_protocol_path,
        expected_protocol_sha256=transfer_protocol_sha256,
    )
    readiness = build_training_readiness(
        lane_root=root,
        training_protocol_path=training_protocol_path,
        expected_training_protocol_sha256=training_protocol_sha256,
        transfer_protocol_path=transfer_protocol_path,
        expected_transfer_protocol_sha256=transfer_protocol_sha256,
    )
    state = {
        "schema_version": "bigan-challenge-model-15m-governance-monitor-state-v1",
        "status": (
            "training_readiness_gate_passed_waiting_for_explicit_training_action"
            if readiness["training_start_allowed"]
            else "monitoring_development_lane_fail_closed"
        ),
        "updated_at": datetime.now(UTC).isoformat(),
        "monitor_pid": os.getpid(),
        "health": {
            "attempted_market_count": health["cumulative"]["attempted_market_count"],
            "quality_valid_market_count": health["cumulative"]["quality_valid_market_count"],
            "quality_valid_outcome_finalized_market_count": health["cumulative"][
                "quality_valid_outcome_finalized_market_count"
            ],
            "paired_executable_ask_coverage": health["cumulative"]["paired_up_down_executable_ask"][
                "coverage"
            ],
        },
        "transfer": transfer,
        "training_readiness": {
            "training_start_allowed": readiness["training_start_allowed"],
            "blocking_reason_codes": readiness["blocking_reason_codes"],
        },
        "attempt_120_authorized": False,
        "model_training_started": False,
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
        "safety": dict(SAFETY),
    }
    atomic_write_json(root / "governance_monitor_state.json", state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-root", required=True)
    parser.add_argument(
        "--transfer-protocol",
        default=str(DEFAULT_TRANSFER_PROTOCOL),
    )
    parser.add_argument("--transfer-protocol-sha256", required=True)
    parser.add_argument(
        "--training-protocol",
        default=str(DEFAULT_TRAINING_PROTOCOL),
    )
    parser.add_argument("--training-protocol-sha256", required=True)
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    parser.add_argument("--run-once", action="store_true")
    args = parser.parse_args()
    state = run_service(
        service_root=args.service_root,
        transfer_protocol_path=args.transfer_protocol,
        transfer_protocol_sha256=args.transfer_protocol_sha256,
        training_protocol_path=args.training_protocol,
        training_protocol_sha256=args.training_protocol_sha256,
        poll_seconds=args.poll_seconds,
        run_once=args.run_once,
    )
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
