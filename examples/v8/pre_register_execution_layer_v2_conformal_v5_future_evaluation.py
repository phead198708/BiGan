"""Freeze the #204 conformal-v5 future evaluation before future target access."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.training.execution_layer_v2_conformal_v5_future_evaluation import (  # noqa: E402
    ConformalV5FuturePreRegistrationConfig,
    pre_register_conformal_v5_future_evaluation,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="examples/v8/polymarket_runs")
    parser.add_argument("--evaluation-profile", required=True)
    parser.add_argument("--evaluation-profile-sha256", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--candidate-manifest-sha256", required=True)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--baseline-manifest-sha256", required=True)
    parser.add_argument("--collector-protocol", required=True)
    parser.add_argument("--collector-protocol-sha256", required=True)
    parser.add_argument("--builder-git-commit", required=True)
    parser.add_argument("--preregistration-created-ts", type=int, required=True)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args(argv)
    result = pre_register_conformal_v5_future_evaluation(
        ConformalV5FuturePreRegistrationConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            evaluation_profile_path=args.evaluation_profile,
            expected_evaluation_profile_sha256=args.evaluation_profile_sha256,
            candidate_manifest_path=args.candidate_manifest,
            expected_candidate_manifest_sha256=args.candidate_manifest_sha256,
            baseline_manifest_path=args.baseline_manifest,
            expected_baseline_manifest_sha256=args.baseline_manifest_sha256,
            collector_protocol_path=args.collector_protocol,
            expected_collector_protocol_sha256=args.collector_protocol_sha256,
            builder_git_commit=args.builder_git_commit,
            preregistration_created_ts=args.preregistration_created_ts,
            overwrite_existing=args.overwrite_existing,
        )
    )
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "pre_registration_ready": result["report"]["pre_registration_ready"],
                "minimum_collection_decision_ts": result["report"][
                    "minimum_collection_decision_ts"
                ],
                "prior_market_count": result["report"]["prior_market_count"],
                "source_boundary_path": str(result["source_boundary_path"]),
                "source_boundary_sha256": result["source_boundary_sha256"],
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
                "future_labels_outcomes_or_pnl_opened": False,
                "prediction_attempted": False,
                "paper_only": True,
                "capital_at_risk": False,
                "v8_execution_handoff_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
