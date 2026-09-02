#!/usr/bin/env python3
"""Freeze the remediation split or evaluate its selected candidate exactly once."""

from __future__ import annotations

import argparse
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_pre_promotion_remediation import (
    evaluate_remediation_candidate_once,
    freeze_remediation_fresh_split,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-dir", required=True, type=Path)
    parser.add_argument("--fresh-corpus-rows-path", type=Path)
    parser.add_argument("--fresh-corpus-quality-report-path", type=Path)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.evaluate:
        result = evaluate_remediation_candidate_once(goal_dir=args.goal_dir)
        print(f"fit_report={result.fit_report_path}")
        print(f"fresh_validation_report={result.validation_report_path}")
        print(f"artifact_eligible={str(result.artifact_eligible).lower()}")
        print(f"frozen_diagnostic_artifact={result.artifact_path}")
    else:
        if not args.fresh_corpus_rows_path or not args.fresh_corpus_quality_report_path:
            parser.error("fresh corpus rows and quality report are required")
        result = freeze_remediation_fresh_split(
            goal_dir=args.goal_dir,
            fresh_corpus_rows_path=args.fresh_corpus_rows_path,
            fresh_corpus_quality_report_path=args.fresh_corpus_quality_report_path,
        )
        print(f"calibration_corpus_manifest={result.corpus_manifest_path}")
        print(f"fresh_split_manifest={result.split_manifest_path}")
        print(
            "fresh_split_manifest_sha256="
            f"{result.split_manifest_sha256_path.read_text(encoding='utf-8').strip()}"
        )
        print(f"split_leakage_report={result.leakage_report_path}")
        print(
            "fresh_validation_gate_passed="
            f"{str(result.fresh_validation_gate_passed).lower()}"
        )
    print("promotion_evidence_stage_started=false")
    print("promotion_evidence_eligible=false")
    print("v8_execution_handoff_allowed=false")


if __name__ == "__main__":
    main()
