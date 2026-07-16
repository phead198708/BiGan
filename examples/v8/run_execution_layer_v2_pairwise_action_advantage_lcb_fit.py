"""Fit and confirm #175 after immutable 90/45/60 role assignment is ready."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.training.execution_layer_v2_pairwise_action_advantage_lcb_fit import (  # noqa: E402
    PairwiseActionAdvantageLCBFitConfig,
    fit_pairwise_action_advantage_lcb,
)

DEFAULT_FEATURE_CONTRACT = ROOT / (
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_pairwise_action_advantage_lcb_feature_contract_v1.json"
)


def run_fit(
    *,
    run_id: str,
    output_dir: Path | str,
    role_assignment_manifest: Path | str,
    role_assignment_manifest_sha256: str,
    feature_contract: Path | str,
    feature_contract_sha256: str,
) -> dict:
    result = fit_pairwise_action_advantage_lcb(
        PairwiseActionAdvantageLCBFitConfig(
            run_id=run_id,
            output_dir=output_dir,
            role_assignment_manifest_path=role_assignment_manifest,
            expected_role_assignment_manifest_sha256=(role_assignment_manifest_sha256),
            feature_contract_path=feature_contract,
            expected_feature_contract_sha256=feature_contract_sha256,
        )
    )
    report = result["validation_report"]
    candidate = report["candidate_metrics"]
    baseline = report["baseline_metrics"]
    confirmatory_started = "confirmatory_gate_passed" in report
    gate_passed = bool(
        report.get("confirmatory_gate_passed")
        if confirmatory_started
        else report.get("development_freeze_gate_passed")
    )
    blockers = list(
        report.get("confirmatory_gate_blocking_reason_codes")
        or report.get("development_freeze_blocking_reason_codes")
        or []
    )
    return {
        "run_id": run_id,
        "development_freeze_gate_passed": report.get("development_freeze_gate_passed"),
        "active_gate_passed": gate_passed,
        "confirmatory_evaluation_started": confirmatory_started,
        "confirmatory_gate_passed": (
            report.get("confirmatory_gate_passed") if confirmatory_started else False
        ),
        "gate_blocking_reason_codes": blockers,
        "candidate_accepted_bet_count": candidate["accepted_bet_count"],
        "candidate_net_pnl": candidate["net_pnl_sum"],
        "candidate_roi": candidate["roi"],
        "baseline_accepted_bet_count": baseline["accepted_bet_count"],
        "baseline_net_pnl": baseline["net_pnl_sum"],
        "candidate_minus_baseline_net_pnl": report["candidate_minus_baseline_net_pnl"],
        "candidate_frozen_for_future_evaluation": report.get(
            "candidate_frozen_for_future_evaluation", False
        ),
        "future_collection_allowed": report.get("future_collection_allowed", False),
        "freeze_manifest_path": str(result["freeze_manifest_path"]),
        "freeze_manifest_sha256": result["freeze_manifest_sha256"],
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "paper_only": True,
        "capital_at_risk": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="examples/v8/polymarket_runs")
    parser.add_argument("--role-assignment-manifest", required=True)
    parser.add_argument("--role-assignment-manifest-sha256", required=True)
    parser.add_argument("--feature-contract", default=str(DEFAULT_FEATURE_CONTRACT))
    parser.add_argument("--feature-contract-sha256", required=True)
    args = parser.parse_args(argv)
    summary = run_fit(
        run_id=args.run_id,
        output_dir=args.output_dir,
        role_assignment_manifest=args.role_assignment_manifest,
        role_assignment_manifest_sha256=args.role_assignment_manifest_sha256,
        feature_contract=args.feature_contract,
        feature_contract_sha256=args.feature_contract_sha256,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["confirmatory_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
