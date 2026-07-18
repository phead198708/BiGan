"""Append one completed collection-only batch to the persistent hash-chain index."""

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
    PersistentOutcomeBlindBatchIndexConfig,
    index_persistent_outcome_blind_batch,
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
    parser.add_argument("--batch-summary", required=True)
    parser.add_argument("--batch-summary-sha256", required=True)
    parser.add_argument("--collector-git-commit", required=True)
    args = parser.parse_args(argv)
    result = index_persistent_outcome_blind_batch(
        PersistentOutcomeBlindBatchIndexConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            protocol_path=args.protocol,
            expected_protocol_sha256=args.protocol_sha256,
            index_path=args.index_path,
            batch_summary_path=args.batch_summary,
            expected_batch_summary_sha256=args.batch_summary_sha256,
            collector_git_commit=args.collector_git_commit,
        )
    )
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "appended_entry_count": result["report"]["appended_entry_count"],
                "index_entry_count": result["report"]["index_entry_count"],
                "quality_valid_index_entry_count": result["report"][
                    "quality_valid_index_entry_count"
                ],
                "index_path": str(result["index_path"]),
                "index_sha256": result["index_sha256"],
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
                "labels_outcomes_or_pnl_opened": False,
                "paper_only": True,
                "capital_at_risk": False,
                "v8_execution_handoff_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
