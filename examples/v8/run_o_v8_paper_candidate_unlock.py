"""Run the #160 O v8 paper-candidate unlock evidence bundle."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from bigan.v8.polymarket.training.o_v8_paper_candidate_unlock import (
    PINNED_ISSUE_159_RUN_ID,
    PolymarketOV8PaperCandidateUnlockConfig,
    run_polymarket_o_v8_paper_candidate_unlock,
)

DEFAULT_ISSUE_159_EVAL_DIR = Path(
    "examples/v8/polymarket_runs/"
    "o-v8-future-holdout-diversified7-eval-20260703T061730Z-20260703T065518Z"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build v8 O paper-candidate unlock and paper-only loop reports."
    )
    parser.add_argument(
        "--run-id",
        default=f"o-v8-paper-candidate-unlock-{_utc_stamp()}",
        help="Run id for the output bundle.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/v8/polymarket_runs"),
        help="Parent output directory.",
    )
    parser.add_argument(
        "--issue-159-eval-dir",
        type=Path,
        default=DEFAULT_ISSUE_159_EVAL_DIR,
        help="Pinned #159 diversified holdout evaluation directory.",
    )
    parser.add_argument(
        "--manual-approval",
        action="store_true",
        help="Explicitly approve local paper-only internal execution loop.",
    )
    parser.add_argument(
        "--manual-approval-id",
        default="issue-160-local-paper-candidate-approval",
        help="Stable manual approval id to hash into the report.",
    )
    parser.add_argument(
        "--manual-approval-operator",
        default="codex",
        help="Operator name recorded in the approval payload.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Replace an existing run directory with the same run id.",
    )
    args = parser.parse_args()

    config = PolymarketOV8PaperCandidateUnlockConfig(
        run_id=args.run_id,
        output_dir=args.output_dir,
        issue_159_eval_dir=args.issue_159_eval_dir,
        manual_approval_approved=args.manual_approval,
        manual_approval_id=args.manual_approval_id,
        manual_approval_operator=args.manual_approval_operator,
        overwrite_existing=args.overwrite_existing,
    )
    result = run_polymarket_o_v8_paper_candidate_unlock(config)
    manifest = result.manifest
    print(f"run_id={args.run_id}")
    print(f"pinned_issue_159_run_id={PINNED_ISSUE_159_RUN_ID}")
    print(f"output_dir={result.output_dir}")
    print(f"paper_candidate_allowed={manifest['paper_candidate_allowed']}")
    print(
        "paper_internal_execution_loop_enabled="
        f"{manifest['paper_internal_execution_loop_enabled']}"
    )
    print(
        "v8_paper_internal_handoff_allowed="
        f"{manifest['v8_paper_internal_handoff_allowed']}"
    )
    print(f"v8_execution_handoff_allowed={manifest['v8_execution_handoff_allowed']}")
    print(f"paper_order_intent_count={manifest['paper_order_intent_count']}")
    print(f"paper_fill_count={manifest['paper_fill_count']}")
    print(f"manifest={result.artifact_paths['manifest']}")
    print(f"manifest_sha256={result.artifact_hashes['manifest']}")


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":
    main()
