#!/usr/bin/env python3
"""Seal the immutable v8 pre-promotion remediation bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_pre_promotion_remediation import (
    ExecutionLayerV2RemediationFinalizationConfig,
    finalize_pre_promotion_remediation_goal,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-dir", required=True, type=Path)
    parser.add_argument("--historical-collection-dir", action="append", default=[])
    parser.add_argument("--outcome-reconciliation-dir", action="append", default=[])
    parser.add_argument("--fresh-corpus-manifest-path", type=Path)
    parser.add_argument("--stop-reason-code", action="append", default=[])
    parser.add_argument("--resumable-next-command")
    args = parser.parse_args()
    result = finalize_pre_promotion_remediation_goal(
        ExecutionLayerV2RemediationFinalizationConfig(
            goal_dir=args.goal_dir,
            historical_collection_dirs=tuple(args.historical_collection_dir),
            outcome_reconciliation_dirs=tuple(args.outcome_reconciliation_dir),
            fresh_corpus_manifest_path=args.fresh_corpus_manifest_path,
            stop_reason_codes=tuple(args.stop_reason_code),
            resumable_next_command=args.resumable_next_command,
        )
    )
    print(f"final_state={result.final_state}")
    print(f"pre_promotion_readiness_report={result.report_path}")
    print(f"pre_promotion_readiness_manifest={result.manifest_path}")
    print(
        "pre_promotion_readiness_manifest_sha256="
        f"{result.manifest_sha256_path.read_text(encoding='utf-8').strip()}"
    )
    print("promotion_evidence_stage_started=false")
    print("promotion_evidence_eligible=false")
    print("live_evidence_stage_started=false")
    print("live_evidence_allowed=false")
    print("v8_execution_handoff_allowed=false")


if __name__ == "__main__":
    main()
