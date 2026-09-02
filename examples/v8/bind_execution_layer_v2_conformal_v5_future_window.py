"""Bind one immutable #192 window to the frozen #204 candidate before prediction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.training.execution_layer_v2_conformal_v5_future_evaluation import (  # noqa: E402
    ConformalV5FutureWindowBindingConfig,
    bind_conformal_v5_future_window_before_prediction,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="examples/v8/polymarket_runs")
    parser.add_argument("--preregistration-manifest", required=True)
    parser.add_argument("--preregistration-manifest-sha256", required=True)
    parser.add_argument("--window-manifest", required=True)
    parser.add_argument("--window-manifest-sha256", required=True)
    parser.add_argument("--builder-git-commit", required=True)
    parser.add_argument("--binding-created-ts", type=int, required=True)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args(argv)
    result = bind_conformal_v5_future_window_before_prediction(
        ConformalV5FutureWindowBindingConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            preregistration_manifest_path=args.preregistration_manifest,
            expected_preregistration_manifest_sha256=(args.preregistration_manifest_sha256),
            window_manifest_path=args.window_manifest,
            expected_window_manifest_sha256=args.window_manifest_sha256,
            builder_git_commit=args.builder_git_commit,
            binding_created_ts=args.binding_created_ts,
            overwrite_existing=args.overwrite_existing,
        )
    )
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "candidate_window_binding_passed": result["report"][
                    "candidate_window_binding_passed"
                ],
                "selected_market_count": result["report"]["selected_market_count"],
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
                "feature_materialization_attempted": False,
                "prediction_attempted": False,
                "future_labels_outcomes_or_pnl_opened": False,
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
