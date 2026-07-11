#!/usr/bin/env python3
"""Seal the v8 Execution Layer v2 pre-promotion readiness evidence bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_pre_promotion_readiness import (
    ExecutionLayerV2PrePromotionFinalizationConfig,
    finalize_pre_promotion_readiness_goal,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-dir", required=True, type=Path)
    parser.add_argument(
        "--historical-collection-dir", action="append", default=[], type=Path
    )
    parser.add_argument(
        "--outcome-reconciliation-dir", action="append", default=[], type=Path
    )
    parser.add_argument("--calibration-corpus-dir", type=Path)
    parser.add_argument("--calibration-run-dir", type=Path)
    parser.add_argument("--stop-reason-code", action="append", default=[])
    parser.add_argument("--resumable-next-command")
    args = parser.parse_args()

    result = finalize_pre_promotion_readiness_goal(
        ExecutionLayerV2PrePromotionFinalizationConfig(
            goal_dir=args.goal_dir,
            historical_collection_dirs=tuple(args.historical_collection_dir),
            outcome_reconciliation_dirs=tuple(args.outcome_reconciliation_dir),
            calibration_corpus_dir=args.calibration_corpus_dir,
            calibration_run_dir=args.calibration_run_dir,
            stop_reason_codes=tuple(args.stop_reason_code),
            resumable_next_command=args.resumable_next_command,
        )
    )
    print(f"goal_dir={result.goal_dir}")
    print(f"final_state={result.final_state}")
    print(
        "pre_promotion_readiness_complete="
        f"{str(result.pre_promotion_readiness_complete).lower()}"
    )
    print(f"readiness_report={result.readiness_report_path}")
    print(f"readiness_manifest={result.readiness_manifest_path}")
    print(
        "readiness_manifest_sha256="
        f"{result.readiness_manifest_sha256_path.read_text(encoding='utf-8').strip()}"
    )
    print("promotion_evidence_stage_started=false")
    print("promotion_evidence_eligible=false")
    print("live_evidence_stage_started=false")
    print("live_evidence_allowed=false")
    print("v8_execution_handoff_allowed=false")


if __name__ == "__main__":
    main()
