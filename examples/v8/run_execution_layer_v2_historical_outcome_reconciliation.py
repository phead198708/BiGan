#!/usr/bin/env python3
"""Resolve historical paper outcomes through the read-only Polymarket CLOB API."""

from __future__ import annotations

import argparse
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_historical_outcome_reconciliation import (
    ExecutionLayerV2HistoricalOutcomeReconciliationConfig,
    run_execution_layer_v2_historical_outcome_reconciliation,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest",
        action="append",
        required=True,
        help="Completed historical paper-run manifest; repeat for multiple runs.",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir",
        default="examples/v8/polymarket_runs",
    )
    parser.add_argument("--request-timeout-seconds", type=float, default=10.0)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--overwrite-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_execution_layer_v2_historical_outcome_reconciliation(
        ExecutionLayerV2HistoricalOutcomeReconciliationConfig(
            run_id=args.run_id,
            source_manifest_paths=tuple(Path(path) for path in args.source_manifest),
            output_dir=Path(args.output_dir),
            request_timeout_seconds=args.request_timeout_seconds,
            max_workers=args.max_workers,
            overwrite_existing=args.overwrite_existing,
        )
    )
    report = result.report
    print(f"run_id={report['run_id']}")
    print(f"output_dir={result.output_dir}")
    print(f"source_run_count={report['source_run_count']}")
    print(f"source_bundle_created_count={report['source_bundle_created_count']}")
    print(f"unresolved_fill_count_before={report['unresolved_fill_count_before']}")
    print(f"resolved_fill_count={report['resolved_fill_count']}")
    print(f"unresolved_fill_count_after={report['unresolved_fill_count_after']}")
    print(f"original_source_artifacts_mutated={report['original_source_artifacts_mutated']}")
    print(f"paper_only={report['paper_only']}")
    print(f"capital_at_risk={report['capital_at_risk']}")
    for name, digest in sorted(result.artifact_hashes.items()):
        print(f"{name}_sha256={digest}")


if __name__ == "__main__":
    main()
