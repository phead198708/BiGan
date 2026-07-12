#!/usr/bin/env python3
"""Run immutable phases of the v8 probability-first estimand workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_estimand_reformulation import (
    EstimandReformulationConfig,
    develop_probability_candidates,
    evaluate_future_unseen_shadow,
    finalize_estimand_reformulation_goal,
    freeze_and_evaluate_validation_round,
    initialize_estimand_reformulation_goal,
    utc_now_iso,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize = subparsers.add_parser("initialize")
    initialize.add_argument("--run-id", required=True)
    initialize.add_argument("--prior-blocked-bundle-dir", required=True, type=Path)
    initialize.add_argument("--inspected-rows-path", required=True, type=Path)
    initialize.add_argument("--output-dir", type=Path, default=Path("examples/v8/polymarket_runs"))
    initialize.add_argument("--repository-root", type=Path, default=Path("."))
    initialize.add_argument("--created-at", default=None)
    develop = subparsers.add_parser("develop")
    develop.add_argument("--goal-dir", required=True, type=Path)
    validate = subparsers.add_parser("validate-round")
    validate.add_argument("--goal-dir", required=True, type=Path)
    validate.add_argument("--round-number", required=True, type=int)
    validate.add_argument("--fresh-rows-path", required=True, type=Path)
    validate.add_argument("--fresh-quality-report-path", type=Path)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--goal-dir", required=True, type=Path)
    finalize.add_argument("--stop-reason-code", action="append", default=[])
    shadow = subparsers.add_parser("shadow")
    shadow.add_argument("--goal-dir", required=True, type=Path)
    shadow.add_argument(
        "--shadow-rows-path", action="append", required=True, type=Path
    )
    args = parser.parse_args()
    if args.command == "initialize":
        result = initialize_estimand_reformulation_goal(
            EstimandReformulationConfig(
                run_id=args.run_id,
                output_dir=args.output_dir,
                repository_root=args.repository_root,
                prior_blocked_bundle_dir=args.prior_blocked_bundle_dir,
                inspected_rows_path=args.inspected_rows_path,
                created_at=args.created_at or utc_now_iso(),
            )
        )
    elif args.command == "develop":
        result = develop_probability_candidates(args.goal_dir)
    elif args.command == "validate-round":
        result = freeze_and_evaluate_validation_round(
            args.goal_dir,
            round_number=args.round_number,
            fresh_rows_path=args.fresh_rows_path,
            fresh_quality_report_path=args.fresh_quality_report_path,
        )
    elif args.command == "shadow":
        result = evaluate_future_unseen_shadow(
            args.goal_dir,
            shadow_rows_paths=tuple(args.shadow_rows_path),
        )
    else:
        result = finalize_estimand_reformulation_goal(
            args.goal_dir, stop_reason_codes=args.stop_reason_code
        )
    print(json.dumps(result, indent=2, default=str, sort_keys=True))
    print("promotion_evidence_stage_started=false")
    print("live_evidence_stage_started=false")
    print("v8_execution_handoff_allowed=false")


if __name__ == "__main__":
    main()
