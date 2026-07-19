"""Freeze the #207 policy-selected conformal v6 protocol before target access."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6 import (  # noqa: E402
    PolicySelectedConformalV6PreRegistrationConfig,
    pre_register_policy_selected_conformal_v6,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="examples/v8/polymarket_runs")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--profile-sha256", required=True)
    parser.add_argument("--issue204-window-manifest", required=True)
    parser.add_argument("--issue204-decision-freeze", required=True)
    parser.add_argument("--issue204-prediction-report", required=True)
    parser.add_argument("--collector-index", required=True)
    parser.add_argument("--collector-index-prefix-sha256", required=True)
    parser.add_argument("--collector-protocol", required=True)
    parser.add_argument("--power-report", required=True)
    parser.add_argument("--power-manifest", required=True)
    parser.add_argument("--builder-git-commit", required=True)
    parser.add_argument("--preregistration-created-ts", type=int, required=True)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args(argv)
    result = pre_register_policy_selected_conformal_v6(
        PolicySelectedConformalV6PreRegistrationConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            profile_path=args.profile,
            expected_profile_sha256=args.profile_sha256,
            issue204_window_manifest_path=args.issue204_window_manifest,
            issue204_decision_freeze_path=args.issue204_decision_freeze,
            issue204_prediction_report_path=args.issue204_prediction_report,
            collector_index_path=args.collector_index,
            expected_collector_index_prefix_sha256=args.collector_index_prefix_sha256,
            collector_protocol_path=args.collector_protocol,
            power_report_path=args.power_report,
            power_manifest_path=args.power_manifest,
            builder_git_commit=args.builder_git_commit,
            preregistration_created_ts=args.preregistration_created_ts,
            overwrite_existing=args.overwrite_existing,
        )
    )
    prefix = result["report"]["collector_index_prefix_summary"]
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "preregistration_passed": result["report"]["preregistration_passed"],
                "target_free_decision_group_count": result["attrition_report"][
                    "decision_group_count"
                ],
                "target_free_selected_action_distribution": result["attrition_report"][
                    "selected_action_distribution"
                ],
                "post_issue204_eligible_market_count": prefix[
                    "eligible_quality_valid_row_count"
                ],
                "development_markets_remaining": prefix["development_markets_remaining"],
                "new_development_target_accessed": False,
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
                "paper_only": True,
                "capital_at_risk": False,
                "paper_candidate_allowed": False,
                "v8_execution_handoff_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
