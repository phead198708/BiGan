#!/usr/bin/env python3
"""Export the outcome-blind slug/two-signal feed for promotion collection."""

from __future__ import annotations

import argparse
from pathlib import Path

from bigan.v8.polymarket.residual_promotion_signal_feed import (
    export_outcome_blind_signal_feed,
    run_signal_feed_monitor,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.once:
        report = export_outcome_blind_signal_feed(
            service_root=args.service_root,
            output_path=args.output,
        )
        print(f"output={args.output.resolve()}")
        print(f"attempts_consumed={report['attempts_consumed']}")
        print(f"quality_valid_match_count={report['quality_valid_match_count']}")
        print(f"content_sha256={report['content_sha256']}")
        print("fresh_outcomes_accessed=false")
        return
    run_signal_feed_monitor(
        service_root=args.service_root,
        output_path=args.output,
        poll_seconds=args.poll_seconds,
    )


if __name__ == "__main__":
    main()
