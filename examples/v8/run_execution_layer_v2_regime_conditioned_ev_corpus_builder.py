#!/usr/bin/env python3
"""Build the deterministic historical calibration corpus for issue #167."""

from __future__ import annotations

import argparse
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_regime_conditioned_ev_corpus import (
    ExecutionLayerV2RegimeConditionedEVCorpusConfig,
    run_execution_layer_v2_regime_conditioned_ev_corpus_builder,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scan completed immutable paper-run manifests, build strict v2 "
            "calibration rows, and emit fail-closed corpus readiness evidence."
        )
    )
    parser.add_argument(
        "--source-root",
        action="append",
        required=True,
        type=Path,
        help="Paper-run root to scan; repeat for multiple roots.",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir", default="examples/v8/polymarket_runs", type=Path
    )
    parser.add_argument("--existing-corpus-manifest", type=Path)
    parser.add_argument("--probability-price-tolerance", default=1e-9, type=float)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()

    try:
        result = run_execution_layer_v2_regime_conditioned_ev_corpus_builder(
            ExecutionLayerV2RegimeConditionedEVCorpusConfig(
                run_id=args.run_id,
                source_roots=tuple(args.source_root),
                output_dir=args.output_dir,
                existing_corpus_manifest=args.existing_corpus_manifest,
                probability_price_tolerance=args.probability_price_tolerance,
                overwrite_existing=args.overwrite_existing,
            )
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    report = result.quality_report
    print(f"run_id={args.run_id}")
    print(f"output_dir={result.output_dir}")
    print(
        "source_manifest_discovered_count="
        f"{report['source_manifest_discovered_count']}"
    )
    print(f"source_run_included_count={report['source_run_included_count']}")
    print(f"source_run_excluded_count={report['source_run_excluded_count']}")
    print(f"eligible_row_count={report['eligible_row_count']}")
    print(f"unique_market_count={report['unique_market_count']}")
    print(
        "exact_duplicate_count="
        f"{report['deduplication']['exact_duplicate_count']}"
    )
    print(
        "conflicting_identity_count="
        f"{report['deduplication']['conflicting_identity_count']}"
    )
    print(
        "incremental_full_rebuild_hash_match="
        f"{str(report['incremental_full_rebuild_hash_match']).lower()}"
    )
    print(
        "minimum_protocol_smoke_passed="
        f"{str(report['minimum_protocol_smoke_passed']).lower()}"
    )
    print(
        "initial_real_calibration_candidate_passed="
        f"{str(report['initial_real_calibration_candidate_passed']).lower()}"
    )
    print(
        "preferred_robust_corpus_passed="
        f"{str(report['preferred_robust_corpus_passed']).lower()}"
    )
    print(f"corpus_sha256={result.manifest['corpus_sha256']}")
    print(f"readiness_blocking_reason_codes={report['readiness_blocking_reason_codes']}")
    print("real_frozen_artifact_created=false")
    print("future_shadow_run_started=false")
    print("paper_only=true")
    print("capital_at_risk=false")
    print("v8_execution_handoff_allowed=false")


if __name__ == "__main__":
    main()
