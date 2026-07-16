#!/usr/bin/env python3
"""Claim one post-batch supervisor and optionally stop the superseded waiter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.training.hybrid_pairwise_collection_supervisor_handoff import (  # noqa: E402
    HybridCollectionSupervisorHandoffConfig,
    ProcessIdentity,
    perform_exclusive_collection_supervisor_handoff,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir",
        default="examples/v8/polymarket_runs",
    )
    parser.add_argument("--claim-path", required=True)
    parser.add_argument("--batch-progress", required=True)
    parser.add_argument("--expected-batch-id", required=True)
    parser.add_argument("--observed-at-ts", required=True, type=int)
    parser.add_argument(
        "--superseded-process",
        required=True,
        metavar="ROLE:PID:COMMAND_SUBSTRING:SCRIPT_PATH:SCRIPT_SHA256",
    )
    parser.add_argument(
        "--successor-process",
        required=True,
        metavar="ROLE:PID:COMMAND_SUBSTRING:SCRIPT_PATH:SCRIPT_SHA256",
    )
    parser.add_argument(
        "--protected-process",
        action="append",
        required=True,
        metavar="ROLE:PID:COMMAND_SUBSTRING[:SCRIPT_PATH:SCRIPT_SHA256]",
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--termination-wait-seconds",
        type=float,
        default=10.0,
    )
    parser.add_argument("--overwrite-existing", action="store_true")
    return parser


def _process(value: str) -> ProcessIdentity:
    fields = value.split(":")
    if len(fields) not in {3, 5}:
        raise ValueError(
            "process identity must use "
            "ROLE:PID:COMMAND_SUBSTRING[:SCRIPT_PATH:SCRIPT_SHA256]"
        )
    return ProcessIdentity(
        role=fields[0],
        pid=int(fields[1]),
        required_command_substring=fields[2],
        script_path=(fields[3] if len(fields) == 5 else None),
        expected_script_sha256=(
            fields[4] if len(fields) == 5 else None
        ),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = perform_exclusive_collection_supervisor_handoff(
        HybridCollectionSupervisorHandoffConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            claim_path=args.claim_path,
            batch_progress_path=args.batch_progress,
            expected_batch_id=args.expected_batch_id,
            observed_at_ts=args.observed_at_ts,
            superseded_supervisor=_process(args.superseded_process),
            successor_supervisor=_process(args.successor_process),
            protected_processes=tuple(
                _process(value) for value in args.protected_process
            ),
            apply_termination=args.apply,
            termination_wait_seconds=args.termination_wait_seconds,
            overwrite_existing=args.overwrite_existing,
        )
    )
    report = result["report"]
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "status": report["status"],
                "handoff_applied": report["handoff_applied"],
                "superseded_supervisor_alive_after": report[
                    "superseded_supervisor_alive_after"
                ],
                "successor_supervisor_alive_after": report[
                    "successor_supervisor_alive_after"
                ],
                "all_protected_processes_alive_after": report[
                    "all_protected_processes_alive_after"
                ],
                "capture_count_before": report["capture_count_before"],
                "capture_count_after": report["capture_count_after"],
                "claim_path": str(result["claim_path"]),
                "claim_sha256": result["claim_sha256"],
                "report_path": str(result["report_path"]),
                "report_sha256": result["report_sha256"],
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
                "paper_only": True,
                "capital_at_risk": False,
                "v8_execution_handoff_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if (not args.apply or report["handoff_applied"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
