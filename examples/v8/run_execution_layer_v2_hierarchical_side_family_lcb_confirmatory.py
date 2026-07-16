"""Run the single frozen #174 fresh confirmatory evaluation."""

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
    HierarchicalSideFamilyLCBConfirmatoryEvaluationConfig,
    evaluate_hierarchical_side_family_lcb_confirmatory,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="examples/v8/polymarket_runs")
    parser.add_argument("--development-candidate-freeze-manifest", required=True)
    parser.add_argument("--development-candidate-freeze-manifest-sha256", required=True)
    parser.add_argument("--confirmatory-assignment-manifest", required=True)
    parser.add_argument("--confirmatory-assignment-manifest-sha256", required=True)
    parser.add_argument("--evaluation-implementation-git-commit", required=True)
    args = parser.parse_args(argv)
    result = evaluate_hierarchical_side_family_lcb_confirmatory(
        HierarchicalSideFamilyLCBConfirmatoryEvaluationConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            development_candidate_freeze_manifest_path=(args.development_candidate_freeze_manifest),
            expected_development_candidate_freeze_manifest_sha256=(
                args.development_candidate_freeze_manifest_sha256
            ),
            confirmatory_assignment_manifest_path=(args.confirmatory_assignment_manifest),
            expected_confirmatory_assignment_manifest_sha256=(
                args.confirmatory_assignment_manifest_sha256
            ),
            evaluation_implementation_git_commit=(args.evaluation_implementation_git_commit),
        )
    )
    report = result["report"]
    summary = {
        "run_id": args.run_id,
        "confirmatory_gate_passed": report["confirmatory_gate_passed"],
        "confirmatory_gate_blocking_reason_codes": report[
            "confirmatory_gate_blocking_reason_codes"
        ],
        "candidate_metrics": report["candidate_metrics"],
        "baseline_metrics": report["baseline_metrics"],
        "candidate_minus_baseline_net_pnl": report["candidate_minus_baseline_net_pnl"],
        "market_bootstrap_interval_95": report["market_robustness_diagnostics"][
            "market_bootstrap_interval_95"
        ],
        "manifest_path": str(result["manifest_path"]),
        "manifest_sha256": result["manifest_sha256"],
        "confirmatory_labels_used_for_tuning": False,
        "candidate_frozen_for_future_evaluation": False,
        "future_collection_allowed": False,
        "paper_only": True,
        "capital_at_risk": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["confirmatory_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
