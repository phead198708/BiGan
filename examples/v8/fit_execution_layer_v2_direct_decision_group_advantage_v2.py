"""Fit the #198 direct decision-group action-advantage v2 candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.training.execution_layer_v2_direct_decision_group_advantage_v2_fit import (  # noqa: E402
    DirectDecisionGroupAdvantageV2FitConfig,
    fit_direct_decision_group_advantage_v2,
)

DEFAULT_PROFILE = ROOT / (
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_direct_decision_group_advantage_v2_fit_profile.json"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="examples/v8/polymarket_runs")
    parser.add_argument("--pre-registration-manifest", required=True)
    parser.add_argument("--pre-registration-manifest-sha256", required=True)
    parser.add_argument("--fit-profile", default=str(DEFAULT_PROFILE))
    parser.add_argument("--fit-profile-sha256", required=True)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args(argv)
    result = fit_direct_decision_group_advantage_v2(
        DirectDecisionGroupAdvantageV2FitConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            pre_registration_manifest_path=args.pre_registration_manifest,
            expected_pre_registration_manifest_sha256=(args.pre_registration_manifest_sha256),
            fit_profile_path=args.fit_profile,
            expected_fit_profile_sha256=args.fit_profile_sha256,
            overwrite_existing=args.overwrite_existing,
        )
    )
    training = result["training_report"]
    viability = result["viability_report"]
    calibration = result["calibration"]
    summary = {
        "run_id": args.run_id,
        "research_candidate_frozen": result["candidate_manifest"]["research_candidate_frozen"],
        "fit_market_count": training["fit_market_count"],
        "materialized_action_row_count": training["materialized_action_row_count"],
        "new_internal_oof_market_count": training["new_internal_oof_market_count"],
        "new_internal_oof_prediction_count": training["new_internal_oof_prediction_count"],
        "duplicate_quantile_boundary_merge_count": calibration[
            "duplicate_quantile_boundary_merge_count"
        ],
        "unreachable_empty_bucket_count": calibration["unreachable_empty_bucket_count"],
        "direct_lcb_passed_action_row_count": viability["direct_lcb_passed_action_row_count"],
        "selected_action_distribution": viability["selected_action_distribution"],
        "execution_guard_evaluated_count": viability["execution_guard_evaluated_count"],
        "execution_guard_allowed_count": viability["execution_guard_allowed_count"],
        "outcome_blind_viability_passed": viability["outcome_blind_viability_passed"],
        "outcome_blind_viability_blocking_reason_codes": viability[
            "outcome_blind_viability_blocking_reason_codes"
        ],
        "candidate_specific_future_evaluation_allowed": False,
        "candidate_manifest_path": str(result["candidate_manifest_path"]),
        "candidate_manifest_sha256": result["candidate_manifest_sha256"],
        "future_labels_remain_sealed": True,
        "future_evaluation_started": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "paper_candidate_allowed": False,
        "paper_only": True,
        "capital_at_risk": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
