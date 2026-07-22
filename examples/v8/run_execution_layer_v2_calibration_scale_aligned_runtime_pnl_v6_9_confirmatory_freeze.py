#!/usr/bin/env python3
"""Freeze the issue #231 v6.9 strictly-later confirmatory window."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_calibration_scale_aligned_runtime_pnl_v6_9_confirmatory import (
    V69ConfirmatoryFreezeConfig,
    freeze_v6_9_confirmatory_window,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--service-root", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--expected-candidate-manifest-sha256", required=True)
    parser.add_argument("--collection-plan", type=Path, required=True)
    parser.add_argument("--expected-collection-plan-sha256", required=True)
    parser.add_argument("--collector-index", type=Path, required=True)
    parser.add_argument("--expected-collector-index-sha256", required=True)
    parser.add_argument("--evaluation-profile", type=Path, required=True)
    parser.add_argument("--expected-evaluation-profile-sha256", required=True)
    parser.add_argument("--implementation-commit")
    parser.add_argument("--decision-freeze-created-ts", type=int)
    parser.add_argument("--overwrite-existing", action="store_true")
    return parser


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> int:
    args = _parser().parse_args()
    head = _head()
    if args.implementation_commit is not None and args.implementation_commit != head:
        raise ValueError("implementation commit does not match current HEAD")
    result = freeze_v6_9_confirmatory_window(
        V69ConfirmatoryFreezeConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            service_root_path=args.service_root,
            candidate_manifest_path=args.candidate_manifest,
            expected_candidate_manifest_sha256=args.expected_candidate_manifest_sha256,
            collection_plan_path=args.collection_plan,
            expected_collection_plan_sha256=args.expected_collection_plan_sha256,
            collector_index_path=args.collector_index,
            expected_collector_index_sha256=args.expected_collector_index_sha256,
            evaluation_profile_path=args.evaluation_profile,
            expected_evaluation_profile_sha256=args.expected_evaluation_profile_sha256,
            implementation_commit=head,
            decision_freeze_created_ts=(
                args.decision_freeze_created_ts or int(time.time() * 1000)
            ),
            overwrite_existing=args.overwrite_existing,
        )
    )
    report = result["report"]
    print(
        json.dumps(
            {
                "run_dir": str(result["run_dir"]),
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
                "selected_window_market_count": report["selected_window_market_count"],
                "guard_accepted_market_count": report["guard_accepted_market_count"],
                "guard_accepted_side_count_diagnostic": report[
                    "guard_accepted_side_count_diagnostic"
                ],
                "target_free_support_gate_passed": report[
                    "target_free_support_gate_passed"
                ],
                "future_target_access_allowed": report["future_target_access_allowed"],
                "labels_outcomes_resolution_or_pnl_opened": False,
                "paper_only": True,
                "paper_candidate_allowed": False,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
