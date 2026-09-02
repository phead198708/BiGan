"""Run one preregistered #226 v6.6 fresh-calibration stage."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_runtime_pnl_v6_6_calibration import (
    PolicySelectedRuntimePNLV66CalibrationConfig,
    run_policy_selected_runtime_pnl_v6_6_calibration,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("freeze_predictions", "settle", "calibrate"),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("examples/v8/polymarket_runs")
    )
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--expected-profile-sha256", required=True)
    parser.add_argument("--point-freeze-manifest", type=Path, required=True)
    parser.add_argument("--expected-point-freeze-manifest-sha256", required=True)
    parser.add_argument("--v6-2-candidate-manifest", type=Path)
    parser.add_argument("--expected-v6-2-candidate-manifest-sha256")
    parser.add_argument("--collector-index", type=Path)
    parser.add_argument("--expected-collector-index-sha256")
    parser.add_argument("--runtime-policy-profile", type=Path)
    parser.add_argument("--expected-runtime-policy-profile-sha256")
    parser.add_argument("--prediction-freeze-manifest", type=Path)
    parser.add_argument("--expected-prediction-freeze-manifest-sha256")
    parser.add_argument("--settled-corpus-index", type=Path)
    parser.add_argument("--expected-settled-corpus-index-sha256")
    parser.add_argument("--implementation-commit")
    parser.add_argument("--stage-started-ts", type=int)
    parser.add_argument("--provider-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--provider-http-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--settlement-max-wait-seconds", type=float, default=600.0)
    parser.add_argument("--settlement-poll-interval-seconds", type=float, default=15.0)
    parser.add_argument("--max-workers", type=int, default=8)
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
    result = run_policy_selected_runtime_pnl_v6_6_calibration(
        PolicySelectedRuntimePNLV66CalibrationConfig(
            stage=args.stage,
            run_id=args.run_id,
            output_dir=args.output_dir,
            profile_path=args.profile,
            expected_profile_sha256=args.expected_profile_sha256,
            point_freeze_manifest_path=args.point_freeze_manifest,
            expected_point_freeze_manifest_sha256=(
                args.expected_point_freeze_manifest_sha256
            ),
            implementation_commit=head,
            v6_2_candidate_manifest_path=args.v6_2_candidate_manifest,
            expected_v6_2_candidate_manifest_sha256=(
                args.expected_v6_2_candidate_manifest_sha256
            ),
            collector_index_path=args.collector_index,
            expected_collector_index_sha256=args.expected_collector_index_sha256,
            runtime_policy_profile_path=args.runtime_policy_profile,
            expected_runtime_policy_profile_sha256=(
                args.expected_runtime_policy_profile_sha256
            ),
            prediction_freeze_manifest_path=args.prediction_freeze_manifest,
            expected_prediction_freeze_manifest_sha256=(
                args.expected_prediction_freeze_manifest_sha256
            ),
            settled_corpus_index_path=args.settled_corpus_index,
            expected_settled_corpus_index_sha256=(
                args.expected_settled_corpus_index_sha256
            ),
            stage_started_ts=args.stage_started_ts or int(time.time() * 1000),
            provider_timeout_seconds=args.provider_timeout_seconds,
            provider_http_timeout_seconds=args.provider_http_timeout_seconds,
            settlement_max_wait_seconds=args.settlement_max_wait_seconds,
            settlement_poll_interval_seconds=args.settlement_poll_interval_seconds,
            max_workers=args.max_workers,
            overwrite_existing=args.overwrite_existing,
        )
    )
    report = result["report"]
    print(
        json.dumps(
            {
                "run_dir": str(result["run_dir"]),
                "manifest_path": str(result["manifest_path"]),
                "stage": args.stage,
                "future_target_access_allowed": report.get(
                    "future_target_access_allowed"
                ),
                "settled_corpus_index_ready": report.get(
                    "settled_corpus_index_ready"
                ),
                "fresh_calibration_gate_passed": report.get(
                    "fresh_calibration_gate_passed"
                ),
                "candidate_scoring_frozen": report.get(
                    "candidate_scoring_frozen"
                ),
                **{
                    key: report[key]
                    for key in (
                        "paper_only",
                        "capital_at_risk",
                        "polymarket_write_enabled",
                        "wallet_signing_enabled",
                        "v8_execution_handoff_allowed",
                        "source_model_candidate_eligible",
                        "freeze_ready",
                        "promotion_evidence_eligible",
                        "#134_resume_allowed",
                        "#146_start_allowed",
                    )
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
