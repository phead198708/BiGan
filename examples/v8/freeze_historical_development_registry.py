#!/usr/bin/env python3
"""Freeze the outcome-blind historical development-market registry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.training.historical_development_registry import (  # noqa: E402
    HistoricalDevelopmentRegistryConfig,
    freeze_historical_development_registry,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="examples/v8/polymarket_runs")
    parser.add_argument("--compatibility-report", required=True)
    parser.add_argument("--compatibility-report-sha256", required=True)
    parser.add_argument("--compatibility-rows", required=True)
    parser.add_argument("--compatibility-rows-sha256", required=True)
    parser.add_argument("--compatibility-manifest", required=True)
    parser.add_argument("--compatibility-manifest-sha256", required=True)
    parser.add_argument("--boundary-freeze-manifest", required=True)
    parser.add_argument("--boundary-freeze-manifest-sha256", required=True)
    parser.add_argument("--selected-market-count", type=int, default=90)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()
    result = freeze_historical_development_registry(
        HistoricalDevelopmentRegistryConfig(
            run_id=args.run_id,
            output_dir=Path(args.output_dir),
            compatibility_report_path=Path(args.compatibility_report),
            expected_compatibility_report_sha256=args.compatibility_report_sha256,
            compatibility_rows_path=Path(args.compatibility_rows),
            expected_compatibility_rows_sha256=args.compatibility_rows_sha256,
            compatibility_manifest_path=Path(args.compatibility_manifest),
            expected_compatibility_manifest_sha256=args.compatibility_manifest_sha256,
            boundary_freeze_manifest_path=Path(args.boundary_freeze_manifest),
            expected_boundary_freeze_manifest_sha256=(
                args.boundary_freeze_manifest_sha256
            ),
            selected_market_count=args.selected_market_count,
            overwrite_existing=args.overwrite_existing,
        )
    )
    report = result["report"]
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "registry_status": report["registry_status"],
                "eligible_pre_boundary_market_count": report[
                    "eligible_pre_boundary_market_count"
                ],
                "selected_market_count": report["selected_market_count"],
                "minimum_selected_decision_ts": report[
                    "minimum_selected_decision_ts"
                ],
                "maximum_selected_decision_ts": report[
                    "maximum_selected_decision_ts"
                ],
                "selected_market_ids_sha256": report[
                    "selected_market_ids_sha256"
                ],
                "report_path": str(result["report_path"]),
                "manifest_path": str(result["manifest_path"]),
                "descriptor_path": str(result["descriptor_path"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
