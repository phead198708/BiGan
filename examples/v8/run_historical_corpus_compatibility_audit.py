#!/usr/bin/env python3
"""Run the outcome-blind historical corpus compatibility audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.training.historical_corpus_compatibility import (  # noqa: E402
    HistoricalCorpusCompatibilityAuditConfig,
    run_historical_corpus_compatibility_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--corpus-root",
        default="/Volumes/PHILIPS/v8/polymarket",
    )
    parser.add_argument(
        "--output-dir",
        default="examples/v8/polymarket_runs",
    )
    parser.add_argument(
        "--protocol-path",
        default=(
            "examples/v8/polymarket_configs/"
            "execution_layer_v2_pairwise_action_advantage_lcb_v1.json"
        ),
    )
    parser.add_argument(
        "--feature-contract-path",
        default=(
            "examples/v8/polymarket_configs/"
            "execution_layer_v2_pairwise_action_advantage_lcb_feature_contract_v1.json"
        ),
    )
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()
    result = run_historical_corpus_compatibility_audit(
        HistoricalCorpusCompatibilityAuditConfig(
            run_id=args.run_id,
            corpus_root=Path(args.corpus_root),
            output_dir=Path(args.output_dir),
            protocol_path=Path(args.protocol_path),
            feature_contract_path=Path(args.feature_contract_path),
            overwrite_existing=args.overwrite_existing,
        )
    )
    report = result["report"]
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "discovered_corpus_count": report["discovered_corpus_count"],
                "unique_market_count": report["unique_market_count"],
                "historical_development_compatible_market_count": report[
                    "historical_development_compatible_market_count"
                ],
                "historical_development_convertible_market_count": report[
                    "historical_development_convertible_market_count"
                ],
                "historical_incompatible_market_count": report[
                    "historical_incompatible_market_count"
                ],
                "estimated_future_hybrid_capture_attempt_count": report[
                    "future_hybrid_protocol_planning_estimate"
                ]["estimated_future_hybrid_capture_attempt_count"],
                "report_path": str(result["report_path"]),
                "manifest_path": str(result["manifest_path"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
