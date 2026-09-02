#!/usr/bin/env python3
"""Run the frozen #199 direct-advantage estimand and support audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_direct_advantage_estimand_audit import (
    DirectAdvantageEstimandAuditConfig,
    run_direct_advantage_estimand_audit,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument(
        "--audit-profile",
        type=Path,
        default=Path(
            "examples/v8/polymarket_configs/"
            "execution_layer_v2_direct_advantage_estimand_audit_profile.json"
        ),
    )
    parser.add_argument("--audit-profile-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/v8/polymarket_runs"))
    parser.add_argument("--overwrite-existing", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_direct_advantage_estimand_audit(
        DirectAdvantageEstimandAuditConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            source_run_dir=args.source_run_dir,
            audit_profile_path=args.audit_profile,
            expected_audit_profile_sha256=args.audit_profile_sha256,
            overwrite_existing=args.overwrite_existing,
        )
    )
    manifest = result["manifest"]
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
                "oracle_best_comparator_hard_gate_recommendation": manifest[
                    "oracle_best_comparator_hard_gate_recommendation"
                ],
                "next_candidate_pre_registration_allowed": manifest[
                    "next_candidate_pre_registration_allowed"
                ],
                "source_model_candidate_eligible": manifest["source_model_candidate_eligible"],
                "promotion_evidence_eligible": manifest["promotion_evidence_eligible"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
