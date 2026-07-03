"""Run a bounded v8 paper-only read-only shadow soak.

This runner uses a deterministic read-only replay feed by default. It writes
heartbeat and periodic-summary artifacts, then feeds Phase 4-compatible paper
decisions into the existing v8 paper harness for Phase 5 and Phase 6 evidence.
It never places real orders, touches real capital, or exposes an exchange write
path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bigan.v8.paper import (  # noqa: E402
    DEFAULT_READONLY_SHADOW_CREATED_AT,
    ReadOnlyShadowSoakConfig,
    run_readonly_shadow_soak,
)

DEFAULT_RUN_ID = "readonly_shadow_short_001"


def run_readonly_shadow_cli(
    output_dir: Path | str,
    *,
    run_id: str = DEFAULT_RUN_ID,
    duration_seconds: int = 300,
    feed_event_interval_seconds: int = 60,
    heartbeat_interval_seconds: int = 10,
    summary_interval_seconds: int = 60,
    inject_degradation: bool = False,
    stop_after_events: int | None = None,
    overwrite_existing: bool = False,
) -> dict[str, object]:
    """Run a deterministic read-only paper shadow soak and return a summary."""

    result = run_readonly_shadow_soak(
        config=ReadOnlyShadowSoakConfig(
            run_id=run_id,
            output_dir=output_dir,
            duration_seconds=duration_seconds,
            feed_event_interval_seconds=feed_event_interval_seconds,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
            summary_interval_seconds=summary_interval_seconds,
            created_at=DEFAULT_READONLY_SHADOW_CREATED_AT,
            inject_degradation=inject_degradation,
            stop_after_events=stop_after_events,
            overwrite_existing=overwrite_existing,
        )
    )
    bundle_path = result.artifact_paths["paper_bundle_manifest"]
    summary_path = result.artifact_paths["paper_run_summary"]
    return {
        "paper_run_summary": str(summary_path),
        "paper_bundle_manifest": str(bundle_path),
        "paper_bundle_manifest_sha256": _sha256_file(bundle_path),
        "run_id": result.run_id,
        "stop_reason": result.final_summary["stop_reason"],
        "feed_event_count": result.final_summary["feed_event_count"],
        "feed_health_passed": result.final_summary["feed_health_passed"],
        "feed_health_reason_codes": result.final_summary[
            "feed_health_reason_codes"
        ],
        "heartbeat_count": result.final_summary["heartbeat_count"],
        "periodic_summary_count": result.final_summary["periodic_summary_count"],
        "paper_only": result.final_summary["paper_only"],
        "capital_at_risk": result.final_summary["capital_at_risk"],
        "broker_exchange_write_enabled": result.final_summary[
            "broker_exchange_write_enabled"
        ],
        "live_exchange_write_enabled": result.final_summary[
            "live_exchange_write_enabled"
        ],
        "phase5_kill_switch_triggered": result.final_summary[
            "phase5_kill_switch_triggered"
        ],
        "phase6_deployment_status": result.final_summary[
            "phase6_deployment_status"
        ],
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "examples" / "v8" / "readonly_shadow_runs",
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument(
        "--duration-seconds",
        type=int,
        default=300,
        help="Bounded deterministic replay duration in seconds.",
    )
    duration.add_argument(
        "--duration-hours",
        type=float,
        help="Bounded deterministic replay duration in hours.",
    )
    parser.add_argument("--feed-event-interval-seconds", type=int, default=60)
    parser.add_argument("--heartbeat-interval-seconds", type=int, default=10)
    parser.add_argument("--summary-interval-seconds", type=int, default=60)
    parser.add_argument(
        "--inject-degradation",
        action="store_true",
        help="Inject deterministic paper degradation for Phase 5 kill-switch validation.",
    )
    parser.add_argument(
        "--stop-after-events",
        type=int,
        default=None,
        help="Create the run STOP file after N feed events to exercise clean shutdown.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Replace an existing run-scoped artifact bundle.",
    )
    args = parser.parse_args(argv)
    duration_seconds = (
        int(args.duration_hours * 60 * 60)
        if args.duration_hours is not None
        else args.duration_seconds
    )
    summary = run_readonly_shadow_cli(
        args.output_dir,
        run_id=args.run_id,
        duration_seconds=duration_seconds,
        feed_event_interval_seconds=args.feed_event_interval_seconds,
        heartbeat_interval_seconds=args.heartbeat_interval_seconds,
        summary_interval_seconds=args.summary_interval_seconds,
        inject_degradation=args.inject_degradation,
        stop_after_events=args.stop_after_events,
        overwrite_existing=args.overwrite_existing,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
