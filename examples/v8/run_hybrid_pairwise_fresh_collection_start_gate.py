#!/usr/bin/env python3
"""Run the explicit #185 hybrid fresh-collection start gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.training.hybrid_pairwise_fresh_collection_roles import (  # noqa: E402
    HybridFreshCollectionStartGateConfig,
    evaluate_hybrid_fresh_collection_start_gate,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir",
        default="examples/v8/polymarket_runs",
    )
    parser.add_argument("--readiness-manifest", required=True)
    parser.add_argument("--readiness-manifest-sha256", required=True)
    parser.add_argument(
        "--collector-script",
        default="examples/v8/run_polymarket_async_round_collector.py",
    )
    parser.add_argument("--collector-script-sha256", required=True)
    parser.add_argument("--collector-git-commit", required=True)
    parser.add_argument("--precollection-freeze-manifest")
    parser.add_argument("--precollection-freeze-manifest-sha256")
    parser.add_argument("--final-prior-quarantine")
    parser.add_argument("--final-prior-quarantine-sha256")
    parser.add_argument("--authorization")
    parser.add_argument("--authorization-sha256")
    parser.add_argument("--overwrite-existing", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_hybrid_fresh_collection_start_gate(
        HybridFreshCollectionStartGateConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            readiness_manifest_path=args.readiness_manifest,
            expected_readiness_manifest_sha256=(
                args.readiness_manifest_sha256
            ),
            collector_script_path=args.collector_script,
            expected_collector_script_sha256=args.collector_script_sha256,
            collector_git_commit=args.collector_git_commit,
            precollection_freeze_manifest_path=(
                args.precollection_freeze_manifest
            ),
            expected_precollection_freeze_manifest_sha256=(
                args.precollection_freeze_manifest_sha256
            ),
            final_prior_quarantine_path=args.final_prior_quarantine,
            expected_final_prior_quarantine_sha256=(
                args.final_prior_quarantine_sha256
            ),
            authorization_path=args.authorization,
            expected_authorization_sha256=args.authorization_sha256,
            overwrite_existing=args.overwrite_existing,
        )
    )
    report = result["report"]
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "status": report["status"],
                "collection_start_allowed": report[
                    "collection_start_allowed"
                ],
                "collection_start_command_generated": report[
                    "collection_start_command_generated"
                ],
                "collector_execution_attempted": False,
                "blocking_reason_codes": report[
                    "blocking_reason_codes"
                ],
                "launch_plan_path": (
                    None
                    if result["launch_plan_path"] is None
                    else str(result["launch_plan_path"])
                ),
                "launch_plan_sha256": result["launch_plan_sha256"],
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
    return 0 if report["collection_start_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
