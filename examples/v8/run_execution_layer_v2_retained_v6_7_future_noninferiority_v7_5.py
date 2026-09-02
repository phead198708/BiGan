"""Run the #238 retained-v6.7 exact-120 future non-inferiority chain."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_retained_v6_7_future_noninferiority_v7_5 import (
    RetainedV67FutureNoninferiorityConfig,
    run_retained_v6_7_future_noninferiority,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("freeze_target_free", "settle", "evaluate_future_pnl"),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/v8/polymarket_runs"))
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--expected-profile-sha256", required=True)
    parser.add_argument("--source-freeze-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-freeze-manifest-sha256", required=True)
    parser.add_argument("--target-free-freeze-manifest", type=Path)
    parser.add_argument("--expected-target-free-freeze-manifest-sha256")
    parser.add_argument("--runtime-policy-profile", type=Path)
    parser.add_argument("--expected-runtime-policy-profile-sha256")
    parser.add_argument("--settled-corpus-index", type=Path)
    parser.add_argument("--expected-settled-corpus-index-sha256")
    parser.add_argument("--provider-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--provider-http-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--settlement-max-wait-seconds", type=float, default=600.0)
    parser.add_argument("--settlement-poll-interval-seconds", type=float, default=15.0)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--implementation-commit")
    parser.add_argument("--stage-started-ts", type=int)
    parser.add_argument("--overwrite-existing", action="store_true")
    return parser.parse_args()


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    args = _parse_args()
    head = _head()
    if args.implementation_commit is not None and args.implementation_commit != head:
        raise ValueError("implementation commit does not match current HEAD")
    result = run_retained_v6_7_future_noninferiority(
        RetainedV67FutureNoninferiorityConfig(
            stage=args.stage,
            run_id=args.run_id,
            output_dir=args.output_dir,
            profile_path=args.profile,
            expected_profile_sha256=args.expected_profile_sha256,
            source_freeze_manifest_path=args.source_freeze_manifest,
            expected_source_freeze_manifest_sha256=(
                args.expected_source_freeze_manifest_sha256
            ),
            target_free_freeze_manifest_path=args.target_free_freeze_manifest,
            expected_target_free_freeze_manifest_sha256=(
                args.expected_target_free_freeze_manifest_sha256
            ),
            runtime_policy_profile_path=args.runtime_policy_profile,
            expected_runtime_policy_profile_sha256=(
                args.expected_runtime_policy_profile_sha256
            ),
            settled_corpus_index_path=args.settled_corpus_index,
            expected_settled_corpus_index_sha256=(
                args.expected_settled_corpus_index_sha256
            ),
            implementation_commit=head,
            stage_started_ts=args.stage_started_ts or int(time.time() * 1000),
            provider_timeout_seconds=args.provider_timeout_seconds,
            provider_http_timeout_seconds=args.provider_http_timeout_seconds,
            settlement_max_wait_seconds=args.settlement_max_wait_seconds,
            settlement_poll_interval_seconds=args.settlement_poll_interval_seconds,
            max_workers=args.max_workers,
            overwrite_existing=args.overwrite_existing,
        )
    )
    print(
        json.dumps(
            {
                "run_dir": str(result["run_dir"]),
                "report_path": str(result["report_path"]),
                "report_sha256": result["report_sha256"],
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
                "index_path": str(result["index_path"]) if result.get("index_path") else None,
                "index_sha256": result.get("index_sha256"),
                "stage": args.stage,
                "comparison_operator": "greater_than_or_equal",
                "equality_passes_noninferiority": True,
                "paper_only": True,
                "capital_at_risk": False,
                "polymarket_write_enabled": False,
                "wallet_signing_enabled": False,
                "v8_execution_handoff_allowed": False,
                "source_model_candidate_eligible": False,
                "freeze_ready": False,
                "promotion_evidence_eligible": False,
                "paper_candidate_allowed": False,
                "#134_resume_allowed": False,
                "#146_start_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
