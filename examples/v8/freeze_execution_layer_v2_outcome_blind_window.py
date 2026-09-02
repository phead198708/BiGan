"""Freeze an immutable strictly-later window from the persistent raw index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.training.execution_layer_v2_persistent_outcome_blind_collector import (  # noqa: E402
    OutcomeBlindWindowFreezeConfig,
    freeze_outcome_blind_window,
)

DEFAULT_PROTOCOL = ROOT / (
    "examples/v8/polymarket_configs/execution_layer_v2_persistent_outcome_blind_collector_v1.json"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="examples/v8/polymarket_runs")
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--index-sha256", required=True)
    parser.add_argument("--source-boundary-manifest", required=True)
    parser.add_argument("--source-boundary-manifest-sha256", required=True)
    parser.add_argument("--target-valid-market-count", type=int, required=True)
    parser.add_argument("--maximum-scan-count", type=int, required=True)
    parser.add_argument("--builder-git-commit", required=True)
    args = parser.parse_args(argv)
    result = freeze_outcome_blind_window(
        OutcomeBlindWindowFreezeConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            protocol_path=args.protocol,
            expected_protocol_sha256=args.protocol_sha256,
            index_path=args.index_path,
            expected_index_sha256=args.index_sha256,
            source_boundary_manifest_path=args.source_boundary_manifest,
            expected_source_boundary_manifest_sha256=(args.source_boundary_manifest_sha256),
            target_valid_market_count=args.target_valid_market_count,
            maximum_scan_count=args.maximum_scan_count,
            builder_git_commit=args.builder_git_commit,
        )
    )
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "window_freeze_ready": result["report"]["window_freeze_ready"],
                "selected_market_count": result["report"]["selected_market_count"],
                "blocking_reason_codes": result["report"]["blocking_reason_codes"],
                "selected_rows_path": str(result["selected_rows_path"]),
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
                "labels_outcomes_or_pnl_opened_for_selection": False,
                "paper_only": True,
                "capital_at_risk": False,
                "v8_execution_handoff_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["report"]["window_freeze_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
