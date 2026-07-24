#!/usr/bin/env python3
"""Run issue #248 historical or target-free canary stage."""

from __future__ import annotations

import argparse
import json

from bigan.v8.polymarket.training.execution_layer_v2_non_risk_abstention_fallback_v8_3 import (
    NonRiskAbstentionFallbackV83CanaryConfig,
    NonRiskAbstentionFallbackV83HistoricalConfig,
    run_non_risk_abstention_fallback_v8_3_canary,
    run_non_risk_abstention_fallback_v8_3_historical_gate,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="stage", required=True)
    historical = subparsers.add_parser("historical")
    _common(historical)
    historical.add_argument("--historical-manifest", required=True)
    historical.add_argument("--historical-manifest-sha256", required=True)
    historical.add_argument("--evaluation-started-ts", type=int, required=True)
    canary = subparsers.add_parser("canary")
    _common(canary)
    canary.add_argument("--historical-gate-manifest", required=True)
    canary.add_argument("--historical-gate-manifest-sha256", required=True)
    canary.add_argument("--issue246-target-free-manifest", required=True)
    canary.add_argument(
        "--issue246-target-free-manifest-sha256", required=True
    )
    canary.add_argument("--canary-started-ts", type=int, required=True)
    args = parser.parse_args()
    if args.stage == "historical":
        result = run_non_risk_abstention_fallback_v8_3_historical_gate(
            NonRiskAbstentionFallbackV83HistoricalConfig(
                run_id=args.run_id,
                output_dir=args.output_dir,
                profile_path=args.profile,
                expected_profile_sha256=args.profile_sha256,
                historical_manifest_path=args.historical_manifest,
                expected_historical_manifest_sha256=(
                    args.historical_manifest_sha256
                ),
                implementation_commit=args.implementation_commit,
                evaluation_started_ts=args.evaluation_started_ts,
                overwrite_existing=args.overwrite_existing,
            )
        )
    else:
        result = run_non_risk_abstention_fallback_v8_3_canary(
            NonRiskAbstentionFallbackV83CanaryConfig(
                run_id=args.run_id,
                output_dir=args.output_dir,
                profile_path=args.profile,
                expected_profile_sha256=args.profile_sha256,
                historical_gate_manifest_path=args.historical_gate_manifest,
                expected_historical_gate_manifest_sha256=(
                    args.historical_gate_manifest_sha256
                ),
                issue246_target_free_manifest_path=(
                    args.issue246_target_free_manifest
                ),
                expected_issue246_target_free_manifest_sha256=(
                    args.issue246_target_free_manifest_sha256
                ),
                implementation_commit=args.implementation_commit,
                canary_started_ts=args.canary_started_ts,
                overwrite_existing=args.overwrite_existing,
            )
        )
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "report": result["report"],
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="examples/v8/polymarket_runs")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--profile-sha256", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--overwrite-existing", action="store_true")


if __name__ == "__main__":
    main()
