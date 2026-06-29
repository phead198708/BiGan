"""Run frozen M selector post-freeze holdout validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.training.post_freeze_holdout import (  # noqa: E402
    PolymarketPostFreezeHoldoutConfig,
    run_polymarket_m_post_freeze_holdout_validation,
)


def run_polymarket_post_freeze_holdout_cli(
    *,
    frozen_model_dir: Path | str,
    frozen_corpus_dir: Path | str,
    holdout_corpus_dir: Path | str,
    output_dir: Path | str,
    run_id: str = "polymarket_m_post_freeze_holdout_validation",
    overwrite_existing: bool = False,
) -> dict:
    result = run_polymarket_m_post_freeze_holdout_validation(
        PolymarketPostFreezeHoldoutConfig(
            frozen_model_dir=frozen_model_dir,
            frozen_corpus_dir=frozen_corpus_dir,
            holdout_corpus_dir=holdout_corpus_dir,
            output_dir=output_dir,
            run_id=run_id,
            overwrite_existing=overwrite_existing,
        )
    )
    report = result.report
    return {
        "run_id": run_id,
        "run_dir": str(result.run_dir),
        "validation_status": report["validation_status"],
        "true_post_freeze_holdout": report["true_post_freeze_holdout"],
        "prediction_attempted": report["prediction_attempted"],
        "selected_entry_count": report["selected_entry_count"],
        "replay_entry_count": report["replay_entry_count"],
        "selected_exit_decision_count": report["selected_exit_decision_count"],
        "replay_entry_reconciliation": report["replay_entry_reconciliation"],
        "replay_total_pnl_sum": report["replay_total_pnl_sum"],
        "holdout_validation_passed": report["holdout_validation_passed"],
        "reason_codes": report["reason_codes"],
        "source_model_candidate_eligible": report["source_model_candidate_eligible"],
        "#146_start_allowed": report["#146_start_allowed"],
        "#134_resume_allowed": report["#134_resume_allowed"],
        "report_path": str(result.artifact_paths["report"]),
        "summary_path": str(result.artifact_paths["summary"]),
        "manifest_path": str(result.artifact_paths["manifest"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-model-dir", required=True)
    parser.add_argument("--frozen-corpus-dir", required=True)
    parser.add_argument("--holdout-corpus-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="polymarket_m_post_freeze_holdout_validation")
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args(argv)

    summary = run_polymarket_post_freeze_holdout_cli(
        frozen_model_dir=args.frozen_model_dir,
        frozen_corpus_dir=args.frozen_corpus_dir,
        holdout_corpus_dir=args.holdout_corpus_dir,
        output_dir=args.output_dir,
        run_id=args.run_id,
        overwrite_existing=args.overwrite_existing,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
