"""Adopt the sealed #227 exact-60 decisions under the #229 v6.8 contract."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_regime_emergent_pnl_v6_8_adoption import (
    V68SealedDecisionAdoptionConfig,
    adopt_sealed_v6_7_decisions_for_v6_8,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("examples/v8/polymarket_runs")
    )
    parser.add_argument("--evaluation-profile", type=Path, required=True)
    parser.add_argument("--expected-evaluation-profile-sha256", required=True)
    parser.add_argument("--parent-prediction-freeze-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-parent-prediction-freeze-manifest-sha256", required=True
    )
    parser.add_argument("--implementation-commit")
    parser.add_argument("--decision-adoption-created-ts", type=int)
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
    result = adopt_sealed_v6_7_decisions_for_v6_8(
        V68SealedDecisionAdoptionConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            evaluation_profile_path=args.evaluation_profile,
            expected_evaluation_profile_sha256=(
                args.expected_evaluation_profile_sha256
            ),
            parent_prediction_freeze_manifest_path=(
                args.parent_prediction_freeze_manifest
            ),
            expected_parent_prediction_freeze_manifest_sha256=(
                args.expected_parent_prediction_freeze_manifest_sha256
            ),
            implementation_commit=head,
            decision_adoption_created_ts=(
                args.decision_adoption_created_ts or int(time.time() * 1000)
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
                "selected_window_market_count": report[
                    "selected_window_market_count"
                ],
                "selected_side_count_diagnostic": report[
                    "selected_side_count_diagnostic"
                ],
                "side_count_hard_gate_enabled": False,
                "future_target_access_allowed": True,
                "labels_outcomes_resolution_or_pnl_opened": False,
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
