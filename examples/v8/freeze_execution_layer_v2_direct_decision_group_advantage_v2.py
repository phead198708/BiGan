"""Freeze the #197 direct decision-group action-advantage v2 protocol."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.training.execution_layer_v2_direct_decision_group_advantage_v2 import (  # noqa: E402
    DirectDecisionGroupAdvantageV2PreRegistrationConfig,
    freeze_direct_decision_group_advantage_v2_pre_registration,
)

DEFAULT_PROTOCOL = ROOT / (
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_direct_decision_group_action_advantage_v2.json"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="examples/v8/polymarket_runs")
    parser.add_argument("--freeze-created-at-ts", type=int)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--role-assignment-manifest", required=True)
    parser.add_argument("--role-assignment-manifest-sha256", required=True)
    parser.add_argument("--power-design", required=True)
    parser.add_argument("--power-design-sha256", required=True)
    parser.add_argument("--power-report", required=True)
    parser.add_argument("--power-report-sha256", required=True)
    parser.add_argument("--issue190-collection-freeze", required=True)
    parser.add_argument("--issue190-collection-freeze-sha256", required=True)
    parser.add_argument("--persistent-collector-protocol", required=True)
    parser.add_argument("--persistent-collector-protocol-sha256", required=True)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args(argv)
    freeze_created_at_ts = args.freeze_created_at_ts or time.time_ns() // 1_000_000
    result = freeze_direct_decision_group_advantage_v2_pre_registration(
        DirectDecisionGroupAdvantageV2PreRegistrationConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            freeze_created_at_ts=freeze_created_at_ts,
            protocol_path=args.protocol,
            expected_protocol_sha256=args.protocol_sha256,
            role_assignment_manifest_path=args.role_assignment_manifest,
            expected_role_assignment_manifest_sha256=(args.role_assignment_manifest_sha256),
            power_design_path=args.power_design,
            expected_power_design_sha256=args.power_design_sha256,
            power_report_path=args.power_report,
            expected_power_report_sha256=args.power_report_sha256,
            issue190_collection_freeze_path=args.issue190_collection_freeze,
            expected_issue190_collection_freeze_sha256=(args.issue190_collection_freeze_sha256),
            persistent_collector_protocol_path=args.persistent_collector_protocol,
            expected_persistent_collector_protocol_sha256=(
                args.persistent_collector_protocol_sha256
            ),
            overwrite_existing=args.overwrite_existing,
        )
    )
    report = result["report"]
    summary = {
        "run_id": args.run_id,
        "pre_registration_ready": report["pre_registration_ready"],
        "protocol_freeze_created_at_ts": report["protocol_freeze_created_at_ts"],
        "minimum_future_decision_ts_exclusive": report["minimum_future_decision_ts_exclusive"],
        "fit_eligible_market_count": report["fit_eligible_market_count"],
        "quarantined_market_count": report["quarantined_market_count"],
        "future_quality_valid_market_target": report["future_quality_valid_market_target"],
        "future_accepted_unique_market_target": report["future_accepted_unique_market_target"],
        "label_outcome_or_pnl_files_opened": report["label_outcome_or_pnl_files_opened"],
        "current_oof_validation_or_confirmatory_pnl_used": report[
            "current_oof_validation_or_confirmatory_pnl_used"
        ],
        "fitting_or_prediction_attempted": report["fitting_or_prediction_attempted"],
        "report_path": str(result["report_path"]),
        "report_sha256": result["report_sha256"],
        "manifest_path": str(result["manifest_path"]),
        "manifest_sha256": result["manifest_sha256"],
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
