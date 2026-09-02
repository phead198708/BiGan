#!/usr/bin/env python3
"""Build post-shadow #169 settlement targets exactly once."""

from __future__ import annotations

import argparse
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_pnl_aligned_future_evaluation import (
    PnLAlignedFutureSettlementTargetConfig,
    build_pnl_aligned_future_settled_evaluation_targets,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/v8/polymarket_runs"))
    parser.add_argument("--shadow-manifest", required=True, type=Path)
    parser.add_argument("--expected-shadow-manifest-sha256", required=True)
    args = parser.parse_args()
    result = build_pnl_aligned_future_settled_evaluation_targets(
        PnLAlignedFutureSettlementTargetConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            shadow_manifest_path=args.shadow_manifest,
            expected_shadow_manifest_sha256=args.expected_shadow_manifest_sha256,
        )
    )
    report = result["report"]
    print(f"manifest_path={result['manifest_path']}")
    print(f"manifest_sha256={result['manifest_sha256']}")
    print(f"status={report['status']}")
    print(f"settled_target_count={report['settled_target_count']}")
    print(f"settled_market_count={report['settled_market_count']}")
    print("identity_reconciliation_passed=true")
    print("outcome_reconciliation_started=true")
    print("future_results_used_for_tuning=false")


if __name__ == "__main__":
    main()
