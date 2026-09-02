#!/usr/bin/env python3
"""Run one explicit #249 settlement or future-PnL evaluation stage."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_non_risk_abstention_fallback_v8_3_future_post_freeze import (
    NonRiskAbstentionFallbackV83FutureEvaluationConfig,
    NonRiskAbstentionFallbackV83FutureSettlementConfig,
    build_non_risk_abstention_fallback_v8_3_future_settled_index,
    evaluate_non_risk_abstention_fallback_v8_3_future_pnl_gate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("settle", "evaluate"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/v8/polymarket_runs"),
    )
    parser.add_argument("--target-free-freeze-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-target-free-freeze-manifest-sha256", required=True
    )
    parser.add_argument("--settled-index", type=Path)
    parser.add_argument("--expected-settled-index-sha256")
    parser.add_argument("--runtime-policy-profile", type=Path)
    parser.add_argument("--expected-runtime-policy-profile-sha256")
    parser.add_argument("--provider-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--provider-http-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--settlement-max-wait-seconds", type=float, default=600.0)
    parser.add_argument(
        "--settlement-poll-interval-seconds", type=float, default=15.0
    )
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--implementation-commit")
    parser.add_argument("--stage-started-ts", type=int)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if args.implementation_commit and args.implementation_commit != head:
        raise ValueError("implementation commit does not match current HEAD")
    stage_ts = args.stage_started_ts or int(time.time() * 1000)
    if args.stage == "settle":
        result = build_non_risk_abstention_fallback_v8_3_future_settled_index(
            NonRiskAbstentionFallbackV83FutureSettlementConfig(
                run_id=args.run_id,
                output_dir=args.output_dir,
                target_free_freeze_manifest_path=(
                    args.target_free_freeze_manifest
                ),
                expected_target_free_freeze_manifest_sha256=(
                    args.expected_target_free_freeze_manifest_sha256
                ),
                implementation_commit=head,
                target_access_started_ts=stage_ts,
                provider_timeout_seconds=args.provider_timeout_seconds,
                provider_http_timeout_seconds=args.provider_http_timeout_seconds,
                settlement_max_wait_seconds=args.settlement_max_wait_seconds,
                settlement_poll_interval_seconds=(
                    args.settlement_poll_interval_seconds
                ),
                max_workers=args.max_workers,
                overwrite_existing=args.overwrite_existing,
            )
        )
    else:
        if (
            args.settled_index is None
            or args.expected_settled_index_sha256 is None
            or args.runtime_policy_profile is None
            or args.expected_runtime_policy_profile_sha256 is None
        ):
            raise ValueError("evaluate requires settled-index and runtime-profile pins")
        result = evaluate_non_risk_abstention_fallback_v8_3_future_pnl_gate(
            NonRiskAbstentionFallbackV83FutureEvaluationConfig(
                run_id=args.run_id,
                output_dir=args.output_dir,
                target_free_freeze_manifest_path=(
                    args.target_free_freeze_manifest
                ),
                expected_target_free_freeze_manifest_sha256=(
                    args.expected_target_free_freeze_manifest_sha256
                ),
                settled_index_path=args.settled_index,
                expected_settled_index_sha256=args.expected_settled_index_sha256,
                runtime_policy_profile_path=args.runtime_policy_profile,
                expected_runtime_policy_profile_sha256=(
                    args.expected_runtime_policy_profile_sha256
                ),
                implementation_commit=head,
                evaluation_started_ts=stage_ts,
                overwrite_existing=args.overwrite_existing,
            )
        )
    report = result["report"]
    print(
        json.dumps(
            {
                "stage": args.stage,
                "run_dir": str(result["run_dir"]),
                "report_path": str(result["report_path"]),
                "report_sha256": result["report_sha256"],
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
                "settled_index_ready": report.get("settled_index_ready"),
                "settled_index_path": (
                    str(result["index_path"]) if result.get("index_path") else None
                ),
                "settled_index_sha256": result.get("index_sha256"),
                "future_pnl_gate_passed": report.get("future_pnl_gate_passed"),
                "future_pnl_gate_blocking_reason_codes": report.get(
                    "future_pnl_gate_blocking_reason_codes"
                ),
                "paper_only": True,
                "capital_at_risk": False,
                "polymarket_write_enabled": False,
                "wallet_signing_enabled": False,
                "v8_execution_handoff_allowed": False,
                "source_model_candidate_eligible": False,
                "freeze_ready": False,
                "promotion_evidence_eligible": False,
                "#134_resume_allowed": False,
                "#146_start_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
