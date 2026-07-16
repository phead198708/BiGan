"""Freeze the #174 development candidate and fresh confirmatory boundary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.training.execution_layer_v2_hierarchical_side_family_lcb import (  # noqa: E402
    HierarchicalSideFamilyLCBFreezeConfig,
    freeze_hierarchical_side_family_lcb_candidate,
)

DEFAULT_PROTOCOL = ROOT / (
    "examples/v8/polymarket_configs/execution_layer_v2_hierarchical_side_family_lcb_v1.json"
)
DEFAULT_FEATURE_CONTRACT = ROOT / (
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_hierarchical_side_family_lcb_feature_contract_v1.json"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="examples/v8/polymarket_runs")
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--feature-contract", default=str(DEFAULT_FEATURE_CONTRACT))
    parser.add_argument("--feature-contract-sha256", required=True)
    parser.add_argument("--issue173-role-assignment-manifest", required=True)
    parser.add_argument("--issue173-role-assignment-manifest-sha256", required=True)
    parser.add_argument("--issue173-development-fit-freeze", required=True)
    parser.add_argument("--issue173-development-fit-freeze-sha256", required=True)
    parser.add_argument("--expected-prior-unique-market-count", type=int, required=True)
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args(argv)
    result = freeze_hierarchical_side_family_lcb_candidate(
        HierarchicalSideFamilyLCBFreezeConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            protocol_path=args.protocol,
            expected_protocol_sha256=args.protocol_sha256,
            feature_contract_path=args.feature_contract,
            expected_feature_contract_sha256=args.feature_contract_sha256,
            issue173_role_assignment_manifest_path=(args.issue173_role_assignment_manifest),
            expected_issue173_role_assignment_manifest_sha256=(
                args.issue173_role_assignment_manifest_sha256
            ),
            issue173_development_fit_freeze_path=(args.issue173_development_fit_freeze),
            expected_issue173_development_fit_freeze_sha256=(
                args.issue173_development_fit_freeze_sha256
            ),
            expected_prior_unique_market_count=(args.expected_prior_unique_market_count),
            git_commit=args.git_commit,
        )
    )
    report = result["training_report"]
    summary = {
        "run_id": args.run_id,
        "development_freeze_gate_passed": report["development_freeze_gate_passed"],
        "development_freeze_blocking_reason_codes": report[
            "development_freeze_blocking_reason_codes"
        ],
        "candidate_metrics": report["candidate_metrics"],
        "baseline_metrics": report["baseline_metrics"],
        "candidate_minus_baseline_net_pnl": report["candidate_minus_baseline_net_pnl"],
        "collection_ready": result["precollection_manifest"]["collection_ready"],
        "development_freeze_manifest_path": str(result["development_freeze_manifest_path"]),
        "development_freeze_manifest_sha256": result["development_freeze_manifest_sha256"],
        "precollection_freeze_manifest_path": str(result["precollection_freeze_manifest_path"]),
        "precollection_freeze_manifest_sha256": result["precollection_freeze_manifest_sha256"],
        "bundle_manifest_path": str(result["bundle_manifest_path"]),
        "bundle_manifest_sha256": result["bundle_manifest_sha256"],
        "paper_only": True,
        "capital_at_risk": False,
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["collection_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
